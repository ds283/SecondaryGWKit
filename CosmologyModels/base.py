from abc import ABC, abstractmethod


class CosmologyBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def H0(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def Hubble(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def d_lnH_dz(self, z: float) -> float:
        raise NotImplementedError
