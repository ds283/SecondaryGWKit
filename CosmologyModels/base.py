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
    def w(self, z: float) -> float:
        raise NotImplementedError

    def epsilon(self, z: float) -> float:
        """
        Evaluate the conventional epsilon parameter eps = -dot(H)/H^2
        :param z: redshift of evaluation
        :return:
        """
        return (1.0 + z) * self.d_lnH_dz(z)
