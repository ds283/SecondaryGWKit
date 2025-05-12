from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS
from CosmologyModels.GenericEOS.QCD_EOS import QCD_EOS
from Units.base import UnitsLike


class QCD_Cosmology(LambdaCDM_GenericEOS):

    def __init__(self, store_id: int, units: UnitsLike, params, max_z: float = 1e14):
        """
        QCD_Cosmology is a convenience wrapper that builds a ParametrizedEOS cosmology using the
        QCD_EOS equation of state
        :param store_id:
        :param units:
        :param params:
        """
        LambdaCDM_GenericEOS.__init__(
            self, store_id, QCD_EOS(units), units, params, max_z=max_z
        )
