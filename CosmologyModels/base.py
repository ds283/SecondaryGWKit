from abc import ABC, abstractmethod

from ray.actor import ActorHandle

from Datastore import DatastoreObject
from Units.base import UnitsLike


class CosmologyBase(DatastoreObject, ABC):
    def __init__(self, store: ActorHandle):
        DatastoreObject.__init__(self, store)
        # no constructor for ABC

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
    def Hubble(self, z: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def d_lnH_dz(self, z: float) -> float:
        raise NotImplementedError
