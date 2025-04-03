from CosmologyModels.LambdaCDM.LambdaCDM import LambdaCDM
from CosmologyModels.QCD.QCDTransition import Gs
from Units import Mpc_units


class QCDCosmology(LambdaCDM):
    """QCD cosmology model that modifies equation of state"""

    def __init__(self, store_id: int, units: Mpc_units, params):
        # inherit _name from parent LambdaCDM instance
        super().__init__(store_id, units, params)

        self.T_CMB = 2.7255  # K
        self.K_to_GeV = 8.6173e-14
        self.cosmology = self

    def T_of_z(self, z: float) -> float:
        """Convert redshift to temperature in GeV with proper g* and g_s factors"""
        T_initial = self.T_CMB * (1 + z) * self.K_to_GeV  # Initial estimate

        # Iterate to find consistent T with g_s factor
        for _ in range(5):  # Usually converges in 2-3 iterations
            g_s_current = Gs(T_initial)
            g_s_cmb = Gs(self.T_CMB * self.K_to_GeV)

            # Handle None values by defaulting to radiation era values
            if g_s_current is None or g_s_cmb is None:
                return T_initial

            g_s_ratio = g_s_cmb / g_s_current
            T_initial = self.T_CMB * (1 + z) * self.K_to_GeV * (g_s_ratio) ** (1 / 3)

        return T_initial

    @property
    def type_id(self) -> int:
        # 1 is the unique ID for the QCDCosmology cosmology type
        return 1
