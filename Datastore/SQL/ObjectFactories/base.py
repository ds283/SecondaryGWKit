from abc import ABC, abstractmethod
from typing import List, Optional


class SQLAFactoryBase(ABC):
    @staticmethod
    @abstractmethod
    def register():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build(payload, conn, table, inserter, tables, inserters):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def store(obj, conn, table, inserter, tables, inserters):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate(obj, conn, table, tables):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_on_startup(conn, table, tables, prune=False):
        raise NotImplementedError

    # Hooks the datastore layer calls (prompts/datastore-generic, prompt 07). They are not
    # abstract, so a factory that has no use for one need not define it.

    @staticmethod
    def revalidate(serial, conn, table, tables) -> bool:
        """
        Recompute the validated flag of the stored row ``serial`` from stored rows alone, write
        it, and return it, by the same rule as ``validate`` applies to an object. A factory whose
        ``register()`` declares ``validated_column`` implements it: the check at open calls it,
        on each shard, after an interrupted validate.
        """
        raise NotImplementedError

    @staticmethod
    def owned_serials(obj) -> Optional[List[Optional[int]]]:
        """
        The serials of the rows ``obj`` owns in another table, in order, or ``None`` if it owns
        none. A replicated store's controlling shard assigns them, and every replica must answer
        with the same; the pool compares the two after a replicated store.
        """
        return None

    @staticmethod
    def inventory_spec():
        """
        How the inventory reads this factory's class (prompts/datastore-generic, prompt 08): an
        ``InventorySpec`` (``Datastore/store_inventory.py``) naming the key's leaf columns, its
        parent references and its association and value tables, or ``None`` where the class is not
        a class of the inventory. It holds no connection and reads nothing; the inventory derives
        which classes it reads, and in what order, from these declarations.
        """
        return None
