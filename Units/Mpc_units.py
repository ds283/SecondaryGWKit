from math import pi, sqrt
from .base import UnitsLike


class Mpc_units(UnitsLike):
    system_name = "Mpc units"

    Mpc = 1.0

    Metre = Mpc / 3.08567758e22
    Kilometre = 1000 * Metre

    sqrt_NewtonG = 1.616199e-35 * Metre

    Kilogram = 1.0 / (2.17651e-8 * sqrt_NewtonG)
    Second = sqrt_NewtonG / 5.39106e-44
    Kelvin = 1.0 / (1.416833e32 * sqrt_NewtonG)

    PlanckMass = sqrt(1.0 / (8.0 * pi)) / sqrt_NewtonG
    eV = PlanckMass / 2.436e27

    c = 299792458 * Metre / Second
