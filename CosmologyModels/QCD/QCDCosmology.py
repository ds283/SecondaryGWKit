import numpy as np

from CosmologyModels.LambdaCDM.LambdaCDM import LambdaCDM
from CosmologyModels.QCD.QCDTransition import w_of_T, G, Gs
from CosmologyModels.model_ids import QCDCOSMOLOGY_IDENTIFIER


class QCDCosmology(LambdaCDM):
    """QCD cosmology model that modifies the equation of state only in the QCD transition range."""

    def __init__(self, store_id: int, units, params):
        super().__init__(store_id, units, params)
        self.T_CMB = 2.7255  # in Kelvin
        self.K_to_GeV = 8.6173e-14
        self.cosmology = self

        # Define the temperature range where the QCD modifications apply.
        # You can adjust these values as needed.
        self.T_min = 0.0054  # GeV, lower bound of valid QCD range
        self.T_max = 1e16  # GeV, upper bound of valid QCD range

    @property
    def type_id(self) -> int:
        # 0 is the unique ID for the LambdaCDM cosmology type
        return QCDCOSMOLOGY_IDENTIFIER

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
            T_initial = (
                self.T_CMB
                * (1 + z)
                * self.K_to_GeV
                * (g_s_cmb / g_s_current) ** (1 / 3)
            )
        return T_initial

    def get_gs(self, T: float) -> float:
        """
        Return the effective entropy degrees of freedom using your QCDTransition module.
        (Assumes the piecewise definitions for Gs(T) are already embedded there.)
        """
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

    ############################################################################

    # Corrected Hubble for QCD transition
    # This method modifies the Hubble function to account for the radiation corrections
    # only in the QCD transition range. Outside that range, it reverts to the standard LCDM Hubble.
    def Hubble(self, z: float) -> float:
        """
        Evaluate H(z) with radiation corrections as follows:
          - For T < T_low: use LCDM Hubble (i.e. no extra corrections).
          - For T_low <= T <= T_switch: use variable corrections based on G(T) and Gs(T).
          - For T > T_switch: use the asymptotic Standard Model values (106.75) for g_* and g_{*,s}.
        """
        T = self.T_of_z(z)
        one_plus_z = 1.0 + z

        # Set threshold values (in GeV); adjust these as needed.
        T_low = 5e-4  # Below this, LCDM behavior applies.
        T_switch = 200  # Above this, use asymptotic SM values.

        if T < T_low:
            # For low T, effective degrees are constant as in today’s universe.
            return super().Hubble(z)
        elif T <= T_switch:
            # In the intermediate (QCD transition) region, use variable effective degrees.
            g_star = G(T)
            gs = Gs(T)
            # Today’s effective numbers (from your document)
            g_star0 = 3.36
            gs0 = 3.94
            rad_factor = (g_star / g_star0) * (gs0 / gs) ** (4 / 3)
        else:
            # For high T, the Standard Model predicts asymptotic values:
            g_star_asympt = 106.75
            gs_asympt = 106.75
            g_star0 = 3.36
            gs0 = 3.94
            rad_factor = (g_star_asympt / g_star0) * (gs0 / gs_asympt) ** (4 / 3)

        rho_r_corr = self.rho_r0 * rad_factor * one_plus_z**4
        rho_m = self.rho_m0 * one_plus_z**3
        rho_total = rho_r_corr + rho_m + self.rho_cc
        H2 = rho_total / (3.0 * self.Mpsq)
        return np.sqrt(H2)
