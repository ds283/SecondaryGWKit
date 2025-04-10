import ray

from CosmologyModels.LambdaCDM import Planck2018
from Datastore.SQL.ShardedPool import ShardedPool
from Units.base import UnitsLike


def build_model_list(pool: ShardedPool, units: UnitsLike):
    params = Planck2018()

    LambdaCDM_Planck2018 = ray.get(
        pool.object_get("LambdaCDM", params=params, units=units)
    )

    QCD_EOS_Planck2018 = ray.get(
        pool.object_get("QCDCosmology", params=params, units=units)
    )

    return [
        {
            "label": "LambdaCDM",
            "cosmology": LambdaCDM_Planck2018,
        },
        {
            "label": "QCDCosmology",
            "cosmology": QCD_EOS_Planck2018,
        },
    ]
