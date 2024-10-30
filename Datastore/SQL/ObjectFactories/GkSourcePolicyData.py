import json
from typing import Optional

import sqlalchemy as sqla
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import GkSourceProxy, GkSourcePolicyData
from CosmologyConcepts import wavenumber_exit_time
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import GkSourcePolicy
from defaults import DEFAULT_STRING_LENGTH

_quality_serialize = {
    "complete": 0,
    "acceptable": 1,
    "marginal": 2,
    "minimal": 3,
    "incomplete": 4,
}

_quality_deserialize = {
    0: "complete",
    1: "acceptable",
    2: "marginal",
    3: "minimal",
    4: "incomplete",
}

_type_serialize = {
    "numeric": 0,
    "WKB": 1,
    "mixed": 2,
}

_type_deserialize = {
    0: "numeric",
    1: "WKB",
    2: "mixed",
}


class sqla_GkSourcePolicyData_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": True,
            "timestamp": True,
            "columns": [
                sqla.Column(
                    "source_serial",
                    sqla.Integer,
                    sqla.ForeignKey("GkSource.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "policy_serial",
                    sqla.Integer,
                    sqla.ForeignKey("GkSourcePolicy.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "wavenumber_exit_serial", sqla.Integer, index=True, nullable=False
                ),
                sqla.Column("type", sqla.SmallInteger, nullable=False),
                sqla.Column("quality", sqla.SmallInteger, nullable=False),
                sqla.Column("crossover_z", sqla.Float(64), nullable=True),
                sqla.Column("Levin_z", sqla.Float(64), nullable=True),
                sqla.Column(
                    "metadata", sqla.String(DEFAULT_STRING_LENGTH), nullable=True
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        source: GkSourceProxy = payload["source"]
        policy: GkSourcePolicy = payload["policy"]
        k_exit: wavenumber_exit_time = payload["k"]

        label: Optional[str] = payload.get("label", None)

        query = sqla.select(
            table.c.serial,
            table.c.type,
            table.c.quality,
            table.c.crossover_z,
            table.c.Levin_z,
            table.c.metadata,
        ).filter(
            table.c.source_serial == source.store_id,
            table.c.policy_serial == policy.store_id,
            table.c.wavenumber_exit_serial == k_exit.store_id,
        )

        try:
            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! GkSourcePolicyData.build(): multiple results found when querying for GkSourcePolicyData"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object
            return GkSourcePolicyData(
                payload=None, source=source, policy=policy, k=k_exit, label=label
            )

        obj = GkSourcePolicyData(
            source=source,
            policy=policy,
            k=k_exit,
            payload={
                "store_id": row_data.serial,
                "type": (
                    _type_deserialize[row_data.type]
                    if row_data.type is not None
                    else None
                ),
                "quality": (
                    _quality_deserialize[row_data.quality]
                    if row_data.quality is not None
                    else None
                ),
                "crossover_z": row_data.crossover_z,
                "Levin_z": row_data.Levin_z,
                "metadata": (
                    json.loads(row_data.metadata)
                    if row_data.metadata is not None
                    else {}
                ),
            },
        )
        obj._deserialized = True

        return obj

    @staticmethod
    def store(obj: GkSourcePolicyData, conn, table, inserter, tables, inserters):
        store_id = inserter(
            conn,
            {
                "source_serial": obj._source_proxy.store_id,
                "policy_serial": obj._policy.store_id,
                "wavenumber_exit_serial": obj._k_exit.store_id,
                "type": _type_serialize[obj._type] if obj._type is not None else None,
                "quality": (
                    _quality_serialize[obj._quality]
                    if obj._quality is not None
                    else None
                ),
                "crossover_z": obj._crossover_z,
                "Levin_z": obj._Levin_z,
                "metadata": (
                    json.dumps(obj._metadata) if obj._metadata is not None else None
                ),
            },
        )

        # set store_id on behalf of the GkSource instance
        obj._my_id = store_id

        return obj
