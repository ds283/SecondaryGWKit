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

    def epsilon(self, z: float) -> float:
        """
        Evaluate the conventional epsilon parameter eps = -dot(H)/H^2
        :param z: redshift of evaluation
        :return:
        """
        one_plus_z = 1.0 + z
        return one_plus_z * self.d_lnH_dz(z)

    def d_epsilon_dz(self, z: float) -> float:
        """
        Evaluate the z derivative of the epsilon parameter
        :param z:
        :return:
        """
        one_plus_z = 1.0 + z
        return self.d_lnH_dz(z) + one_plus_z * self.d2_lnH_dz2(z)

    def d2_epsilon_dz2(self, z: float) -> float:
        """
        Evaluate the 2nd z derivative of the epsilon parameter
        :param z:
        :return:
        """
        one_plus_z = 1.0 + z
        return 2.0 * self.d2_lnH_dz2(z) + one_plus_z * self.d3_dlnH_dz3(z)
