from CosmologyModels.LambdaCDM.LambdaCDM import LambdaCDM
from CosmologyModels.QCD.QCDTransition import w_of_T
import numpy as np

class QCDCosmology(LambdaCDM):
    """QCD cosmology model that modifies the equation of state only in the QCD transition range."""
    
    def __init__(self, store_id: int, units, params):
        super().__init__(store_id, units, params)
        self.T_CMB = 2.7255  # in Kelvin
        self.K_to_GeV = 8.6173e-14
        self.cosmology = self
        
        # Define the temperature range where the QCD modifications apply.
        # You can adjust these values as needed.
        self.T_min = 0.0054   # GeV, lower bound of valid QCD range
        self.T_max = 1e16   # GeV, upper bound of valid QCD range

    def T_of_z(self, z: float) -> float:
        """
        Convert redshift to temperature in GeV accounting for the g_s factors using an iterative method.
        """
        T_initial = self.T_CMB * (1 + z) * self.K_to_GeV
        for _ in range(5):  # Iterate until convergence
            g_s_current = self.get_gs(T_initial)
            g_s_cmb = self.get_gs(self.T_CMB * self.K_to_GeV)
            if g_s_current is None or g_s_cmb is None:
                return T_initial
            T_initial = self.T_CMB * (1 + z) * self.K_to_GeV * (g_s_cmb / g_s_current)**(1/3)
        return T_initial

    def get_gs(self, T: float) -> float:
        """
        Return the effective entropy degrees of freedom using your QCDTransition module.
        (Assumes the piecewise definitions for Gs(T) are already embedded there.)
        """
        from CosmologyModels.QCD.QCDTransition import Gs
        return Gs(T)

    def w(self, z: float) -> float:
        """
        Return the effective equation of state parameter at redshift z.
        If T(z) is within the QCD transition range, use the QCD EoS;
        otherwise, revert to the LCDM wBackground (or a radiation-like value).
        """
        T = self.T_of_z(z)
        if self.T_min <= T <= self.T_max:
            # In the QCD valid temperature range, use the QCD EoS.
            return w_of_T(T)
        else:
            # Outside the QCD range, revert to the default LCDM value.
            return self.wBackground(z)
