import ray
from ray import ObjectRef

from ComputeTargets import BackgroundModel


class ModelProxy:
    def __init__(self, model: BackgroundModel):
        self._ref: ObjectRef = ray.put(model)
        self._store_id: int = model.store()

    @property
    def store_id(self) -> int:
        return self._store_id

    @property
    def get(self) -> BackgroundModel:
        """
        The return value should only be held locally and not persisted, otherwise the entire
        BackgroundModel instance may be serializd when it is passed around by Ray.
        That would defeat the purpose of the proxy.
        :return:
        """
        return ray.get(self._ref)
