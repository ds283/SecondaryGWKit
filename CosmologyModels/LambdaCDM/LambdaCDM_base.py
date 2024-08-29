class LambdaCDM:
    def __init__(self, params, units):
        self.params = params
        self.units = units

        # unpack details of the parameter block so we can access them without extensive nesting
        self.name = params.name

        self.omega_cc = params.omega_cc
        self.omega_m = params.omega_m
        self.f_baryon = params.f_baryon
        self.h = params.h
        self.T_CMB_Kelvin = params.T_CMB_Kelvin
        self.Neff = params.Neff

        # derived quantities
        self.H0 = 100.0 * params.h * units.Kilometre
        self.T_CMB = params.T_CMB_Kelvin * units.Kelvin
