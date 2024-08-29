from abc import ABC, abstractmethod


class UnitsLike(ABC):
    @property
    @abstractmethod
    def system_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Metre(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Kilometre(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Kilogram(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Second(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Kelvin(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def PlanckMass(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def eV(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def c(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Mpc(self):
        raise NotImplementedError
