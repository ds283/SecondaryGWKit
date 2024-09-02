from abc import ABC, abstractmethod


class UnitsLike(ABC):
    def __init__(self, name: str):
        self._name = name
        self._name_hash = hash(name)

    def __eq__(self, other):
        return self._name_hash == other._name_hash

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def system_name(self):
        return self._name

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
