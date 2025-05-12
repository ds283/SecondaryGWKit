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
    def keV(self):
        return 1e3 * self.eV

    @property
    def MeV(self):
        return 1e6 * self.eV

    @property
    def GeV(self):
        return 1e9 * self.eV

    @property
    @abstractmethod
    def c(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def Mpc(self):
        raise NotImplementedError


def check_units(A, B):
    """
    Check that objects A and B are defined with the same units.
    Assumes they both provide a .units property that returns a UnitsLike object
    :param A:
    :param B:
    :return:
    """
    if A.units != B.units:
        raise RuntimeError("Units used for wavenumber k and cosmology are not equal")
