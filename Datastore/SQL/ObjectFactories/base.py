from abc import ABC, abstractmethod


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
