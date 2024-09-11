from abc import ABC, abstractmethod


class SQLAFactoryBase(ABC):
    @staticmethod
    @abstractmethod
    def build(payload, conn, table, inserter, tables, inserters):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate_columns():
        raise NotImplementedError
