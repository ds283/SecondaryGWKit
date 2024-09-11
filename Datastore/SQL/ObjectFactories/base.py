from abc import ABC, abstractmethod


class SQLAFactoryBase(ABC):
    @staticmethod
    @abstractmethod
    def build(payload, engine, table, inserter, tables, inserters):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate_columns():
        raise NotImplementedError
