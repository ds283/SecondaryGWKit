from abc import ABC, abstractmethod


class SQLAFactoryBase(ABC):
    @staticmethod
    @abstractmethod
    async def build(
        payload,
        engine,
        conn,
        table,
        full_query,
        serial_query,
        inserter,
        tables,
        inserters,
    ):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    async def generate_columns():
        raise NotImplementedError
