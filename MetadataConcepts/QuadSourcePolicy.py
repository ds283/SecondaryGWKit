from typing import Optional

from Datastore import DatastoreObject
from defaults import DEFAULT_LEVIN_THRESHOLD, DEFAULT_QUADSOURCE_NUMERIC_POLICY

_allowed_numeric_policies = ["maximize_numeric", "maximize_Levin"]


class QuadSourcePolicy(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        Levin_threshold: int = DEFAULT_LEVIN_THRESHOLD,
        numeric_policy: str = DEFAULT_QUADSOURCE_NUMERIC_POLICY,
        label: Optional[str] = None,
    ):
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        self._label = label

        self._Levin_threshold = Levin_threshold

        if numeric_policy not in _allowed_numeric_policies:
            numeric_policy = DEFAULT_QUADSOURCE_NUMERIC_POLICY
            print(
                f'!! Unknown Quadsource numeric policy "{numeric_policy}". Defaulting to policy "{DEFAULT_QUADSOURCE_NUMERIC_POLICY}"'
            )
        self._numeric_policy = numeric_policy

    @property
    def Levin_threshold(self) -> float:
        return self._Levin_threshold

    @property
    def numeric_policy(self) -> str:
        return self._numeric_policy

    @property
    def label(self) -> Optional[str]:
        return self._label
