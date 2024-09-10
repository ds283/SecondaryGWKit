from abc import ABC, abstractmethod


class SQLAFactoryBase(ABC):
    @staticmethod
    @abstractmethod
    async def build(
        engine, conn, table, full_query, serial_query, tables, inserter, payload
    ):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    async def generate_columns():
        raise NotImplementedError
