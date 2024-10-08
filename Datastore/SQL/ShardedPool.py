import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import ray
import sqlalchemy as sqla

from CosmologyConcepts import wavenumber, wavenumber_exit_time
from Datastore.SQL import Datastore
from Datastore.SQL.Datastore import PathType
from Datastore.SQL.ProfileAgent import ProfileAgent
from Datastore.SQL.SerialPoolBroker import SerialPoolBroker
from MetadataConcepts import version
from Units.base import UnitsLike
from defaults import DEFAULT_STRING_LENGTH

_replicate_tables = [
    "version",
    "store_tag",
    "redshift",
    "wavenumber",
    "tolerance",
    "LambdaCDM",
    "IntegrationSolver",
    "BackgroundModel",
    "BackgroundModelValue",
]
_shard_tables = {
    "wavenumber_exit_time": "k",
    "TkNumericalIntegration": "k",
    "TkNumericalValue": "k",
    "GkNumericalIntegration": "k",
    "GkNumericalValue": "k",
    "TensorSource": "q",
    "TensorSourceValue": "q",
    "GkWKBIntegration": "k",
    "GkWKBValue": "k",
    "GkSource": "k",
    "GkSourceValue": "k",
}


class ShardedPool:
    """
    ShardedPool manages a pool of datastore actors that cooperate to
    form a sharded SQL database
    """

    def __init__(
        self,
        version_label: str,
        db_name: PathType,
        timeout=None,
        shards=10,
        profile_db: Optional[PathType] = None,
        job_name: Optional[str] = None,
    ):
        """
        Initialize a pool of datastore actors
        :param version_label:
        """
        self._job_name = job_name
        self._version_label = version_label

        self._db_name = db_name
        self._timeout = timeout
        self._shards = max(shards, 1)

        # resolve concerts the supplied db_name to an absolute path, resolving symlinks if necessary
        # this database file will be taken to be the primary database
        self._primary_file = Path(db_name).resolve()

        self._shard_db_files = {}
        self._wavenumber_keys = {}

        self._broker = SerialPoolBroker.options(name="SerialPoolBroker").remote(
            name="SerialPoolBroker"
        )

        if profile_db is not None:
            if self._job_name is not None:
                label = f'{self._version_label}-jobname-"{self._job_name}"-primarydb-"{str(db_name)}"-shards-{str(shards)}-{datetime.now().replace(microsecond=0).isoformat()}'
            else:
                label = f'{self._version_label}-primarydb-"{str(db_name)}"-shards-{str(shards)}-{datetime.now().replace(microsecond=0).isoformat()}'

            self._profile_agent = ProfileAgent.options(name="ProfileAgent").remote(
                db_name=profile_db,
                timeout=self._timeout,
                label=label,
            )
        else:
            self._profile_agent = None

        # if primary file is absent, all shard databases should be likewise absent
        if self._primary_file.is_dir():
            raise RuntimeError(
                f'Specified database file "{str(self._db_file)}" is a directory'
            )
        if not self._primary_file.exists():
            # ensure parent directories also exist
            self._primary_file.parents[0].mkdir(exist_ok=True, parents=True)

            stem = self._primary_file.stem
            for i in range(self._shards):
                shard_stem = f"{stem}-shard{i:04d}"
                shard_file = self._primary_file.with_stem(shard_stem)

                if shard_file.exists():
                    raise RuntimeError(
                        f'Primary database is missing, but shard "{str(shard_file)}" already exists'
                    )

                self._shard_db_files[i] = shard_file

            self._create_engine()
            self._write_shard_data()

            print(
                f'>> Created sharded datastore "{str(self._primary_file)}" with {self._shards} shards'
            )

        # otherwise, if primary exists, try to read in shard configuration from it.
        # Then, all shard databases must be present
        else:
            self._create_engine()
            self._read_shard_data()

            num_shards = len(self._shard_db_files)
            print(
                f'>> Opened existing sharded datastore "{str(self._primary_file)}" with {num_shards} shards'
            )

            if num_shards == 0:
                raise RuntimeError(
                    "No shard records were read from the sharded datastore"
                )
            if num_shards != self._shards:
                print(
                    f"!! WARNING: number of shards read from database (={num_shards}) does not match specified number of shards (={self._shards})"
                )

        # create actor pool of datastores, one for each shard
        # we read the version serial number from the first shard that we create
        shard_ids = list(self._shard_db_files.keys())

        shard0_key = shard_ids.pop()
        shard0_file = self._shard_db_files[shard0_key]

        # create the first shard datastore
        shard0_store = Datastore.options(name=f"shard{shard0_key:04d}-store").remote(
            version_label=version_label,
            db_name=shard0_file,
            timeout=self._timeout,
            my_name=f"shard{shard0_key:04d}-store",
            serial_broker=self._broker,
            profile_agent=self._profile_agent,
        )
        self._shards = {shard0_key: shard0_store}

        # get the version label from this store
        self._version = ray.get(
            shard0_store.object_get.remote(version, label=version_label)
        )

        # populate the remaining pool of shard stores
        self._shards.update(
            {
                key: Datastore.options(name=f"shard{key:04d}-store").remote(
                    version_label=version_label,
                    version_serial=self._version.store_id,
                    db_name=self._shard_db_files[key],
                    timeout=self._timeout,
                    my_name=f"shard{key:04d}-store",
                    serial_broker=self._broker,
                    profile_agent=self._profile_agent,
                )
                for key in shard_ids
            }
        )

        # query a list of largest serial numbers from each shard, and notify these to the broker actor
        max_serial_data = ray.get(
            [shard.read_largest_store_ids.remote() for shard in self._shards.values()]
        )
        ray.get(
            [
                self._broker.notify_largest_store_ids.remote(payload)
                for payload in max_serial_data
            ]
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ray.get(
            [
                shard.__exit__.remote(exc_type=None, exc_val=None, exc_tb=None)
                for shard in self._shards.values()
            ]
        )

        if self._profile_agent is not None:
            ray.get(self._profile_agent.clean_up.remote())

        if self._engine is not None:
            self._engine.dispose()

    def _create_engine(self):
        connect_args = {}
        if self._timeout is not None:
            connect_args["timeout"] = self._timeout

        self._engine = sqla.create_engine(
            f"sqlite:///{self._db_name}",
            future=True,
            connect_args=connect_args,
        )
        self._metadata = sqla.MetaData()

        self._shard_file_table = sqla.Table(
            "shards",
            self._metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True, nullable=False),
            sqla.Column("filename", sqla.String(DEFAULT_STRING_LENGTH), nullable=False),
        )
        self._shard_key_table = sqla.Table(
            "shard_keys",
            self._metadata,
            sqla.Column(
                "wavenumber_serial", sqla.Integer, primary_key=True, nullable=False
            ),
            sqla.Column(
                "shard_id",
                sqla.Integer,
                sqla.ForeignKey("shards.serial"),
                nullable=False,
            ),
        )

    def _write_shard_data(self):
        self._shard_file_table.create(self._engine)
        self._shard_key_table.create(self._engine)

        with self._engine.begin() as conn:
            values = [
                {"serial": key, "filename": str(db_name)}
                for key, db_name in self._shard_db_files.items()
            ]

            conn.execute(sqla.insert(self._shard_file_table), values)
            conn.commit()

    def _read_shard_data(self):
        with self._engine.begin() as conn:
            rows = conn.execute(
                sqla.select(
                    self._shard_file_table.c.serial,
                    self._shard_file_table.c.filename,
                )
            )

            for row in rows:
                serial = row.serial
                filename = Path(row.filename)

                if serial in self._shard_db_files:
                    raise RuntimeError(
                        f'Shard #{serial} already exists (database file="{str(filename)}", existing file="{str(self._shard_db_files[serial])}")'
                    )

                self._shard_db_files[row.serial] = Path(row.filename)

            keys = conn.execute(
                sqla.select(
                    self._shard_key_table.c.wavenumber_serial,
                    self._shard_key_table.c.shard_id,
                )
            )

            for key in keys:
                self._wavenumber_keys[key.wavenumber_serial] = key.shard_id

    def object_get(self, ObjectClass, **kwargs):
        if isinstance(ObjectClass, str):
            cls_name = ObjectClass
        else:
            cls_name = ObjectClass.__name__

        if cls_name in _replicate_tables:
            return self._get_impl_replicated_table(cls_name, kwargs)

        if cls_name in _shard_tables.keys():
            return self._get_impl_sharded_table(cls_name, kwargs)

        raise RuntimeError(
            f'Unable to dispatch object_get() for item of type "{cls_name}"'
        )

    def _get_impl_replicated_table(self, cls_name, kwargs):
        # pick a shard id at random to be the "controlling" shard.
        # we will push an initial 'get' to this controlling shard.
        # if a new database object was created by the get, we then have to push a replica
        # to all the other shards
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        # for replicated tables, we should query/insert into *one* datastore, and then enforce
        # that all other datastores get the same store_id; here, there is no need to use our internal
        # information about the next-allocated store_id, and in fact doing so would make the logic
        # here much more complicated. So we avoid that.
        ref = self._shards[shard_key].object_get.remote(cls_name, **kwargs)
        objects = ray.get(ref)

        # was this a vectorized get?
        if "payload_data" in kwargs:
            payload_data = kwargs["payload_data"]

            if len(payload_data) != len(objects):
                raise RuntimeError(
                    f"object_get() data returned from selected datastore (shared={shard_key}) has a different length (length={len(objects)}) to payload data (length={len(payload_data)})"
                )

            # add explicit serial specifier
            new_payload = []
            for i in range(len(payload_data)):
                # if this object has a valid store_id and was not deserialized, push to all the
                # remaining shards in order to replicate it
                if (
                    hasattr(objects[i], "_my_id")
                    and objects[i]._my_id is not None
                    and not hasattr(objects[i], "_deserialized")
                ):
                    payload_data[i]["serial"] = objects[i].store_id
                    new_payload.append(payload_data[i])

            # queue work items to replicate each object in all other shards (recall that shard_key has already been popped from shard_ids,
            # so there is no double insertion here)
            ray.get(
                [
                    self._shards[key].object_get.remote(
                        cls_name, payload_data=new_payload
                    )
                    for key in shard_ids
                ]
            )
        else:
            # this was a scalar get

            # if this object has a valid store_id and was not deserialized, push to all the
            # remaining shards in order to replicate it
            if (
                hasattr(objects, "_my_id")
                and objects._my_id is not None
                and not hasattr(objects, "_deserialized")
            ):
                ray.get(
                    [
                        self._shards[key].object_get.remote(
                            cls_name, serial=objects.store_id, **kwargs
                        )
                        for key in shard_ids
                    ]
                )

        # test whether this query was for a shard key, and, if so, assign any shard keys
        # that are missing
        if cls_name == "wavenumber":
            self._assign_shard_keys(objects)

        # return original object (we just discard any copies returned from other shards)
        return ref

    def _get_impl_sharded_table(self, cls_name, kwargs):
        # for sharded tables, we should query/insert into only the appropriate shard
        shard_key_field = _shard_tables[cls_name]

        # is this a vectorized get?
        if "payload_data" in kwargs:
            payload_data = kwargs["payload_data"]

            work_refs = []
            for item in payload_data:
                k = item[shard_key_field]
                shard_id = self._wavenumber_keys[self._get_k_store_id(k)]

                work_refs.append(
                    self._shards[shard_id].object_get.remote(cls_name, **item)
                )

            return work_refs
            # TODO: consider consolidating all objects for the same shard into a list, for efficiency

        # otherwise, can assume this is scalar get
        k = kwargs[shard_key_field]
        shard_id = self._wavenumber_keys[self._get_k_store_id(k)]

        return self._shards[shard_id].object_get.remote(cls_name, **kwargs)

    def object_store(self, objects):
        if isinstance(objects, list) or isinstance(objects, tuple):
            payload_data = objects
            scalar = False
        else:
            payload_data = [objects]
            scalar = True

        work_refs = []
        for item in payload_data:
            cls_name = type(item).__name__

            if cls_name in _replicate_tables:
                work_refs.extend(self._store_impl_replicated_table(cls_name, item))
                continue

            if cls_name in _shard_tables.keys():
                work_refs.extend(self._store_impl_sharded_table(cls_name, item))
                continue

            raise RuntimeError(
                f'Unable to dispatch object_get() for item of type "{cls_name}"'
            )

        if scalar:
            return work_refs[0]

        return work_refs

    def _store_impl_replicated_table(self, cls_name, item):
        # pick a shard id at random to be the "controlling" shard
        # we will push an initial 'store' to this controlling shard.
        # if a new database object was created by the get, we then have to push a replica
        # to all the other shards
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        ref = self._shards[shard_key].object_store.remote(item)
        object = ray.get(ref)

        # now push the object, complete with its new 'store_id', to all the other shards
        if not hasattr(object, "_my_id") or object._my_id is None:
            raise RuntimeError(
                f'Stored object of type "{cls_name}" was not assigned a store_id field'
            )

        ray.get([self._shards[key].object_store.remote(object) for key in shard_ids])

        return [ref]

    def _store_impl_sharded_table(self, cls_name, item):
        # item need only be pushed to a single shard
        # unlike the replicated case,
        # we don't have to care about what happens to its store_id

        shard_key_field = _shard_tables[cls_name]
        if not hasattr(item, shard_key_field):
            raise RuntimeError(
                f'Unable to determine shard, because object of type "{cls_name}" has no "{shard_key_field}" attribute'
            )

        k = getattr(item, shard_key_field)
        shard_id = self._wavenumber_keys[self._get_k_store_id(k)]

        # TODO: consider consolidating all stores for the same shard into a list, for efficiency
        return [self._shards[shard_id].object_store.remote(item)]

    def object_validate(self, objects):
        # we only expect to call object_store on sharded objects
        if isinstance(objects, list) or isinstance(objects, tuple):
            payload_data = objects
            scalar = False
        else:
            payload_data = [objects]
            scalar = True

        work_refs = []
        for item in payload_data:
            cls_name = type(item).__name__

            if cls_name in _replicate_tables:
                work_refs.extend(self._validate_impl_replicated_table(cls_name, item))
                continue

            if cls_name in _shard_tables.keys():
                work_refs.extend(self._validate_impl_sharded_table(cls_name, item))
                continue

            raise RuntimeError(
                f'Unable to dispatch object_validate() for item of type "{cls_name}"'
            )

        if scalar:
            return work_refs[0]

        return work_refs

    def _validate_impl_replicated_table(self, cls_name, item):
        # pick a shard id at random to be the "controlling" shard
        # we will push an initial 'validate' to this controlling shard.
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        ref = self._shards[shard_key].object_validate.remote(item)
        outcome = ray.get(ref)

        # if object did not validate, do not push validation requests to remaining shards
        if outcome is False or outcome is None:
            return [ref]

        outcomes = ray.get(
            [self._shards[key].object_validate.remote(item) for key in shard_ids]
        )
        if any(oc is not True for oc in outcomes):
            print(f"!! Validation outcomes did not agree between shards:")
            print(f"|    outcomes = {outcomes}")
            raise RuntimeError(
                f'Object validation produced different outcomes on different shards for replicated object of type "{cls_name}"'
            )

        return [ref]

    def _validate_impl_sharded_table(self, cls_name, item):
        # item need only be validated on a single shard
        shard_key_field = _shard_tables[cls_name]
        if not hasattr(item, shard_key_field):
            raise RuntimeError(
                f'Unable to determine shard, because object of type "{cls_name}" has no "{shard_key_field}" attribute'
            )

        k = getattr(item, shard_key_field)
        shard_id = self._wavenumber_keys[self._get_k_store_id(k)]

        # TODO: consider consolidating all validates for the same shard into a list, for efficiency
        return [self._shards[shard_id].object_validate.remote(item)]

    def _get_k_store_id(self, obj):
        if isinstance(obj, wavenumber):
            return obj.store_id

        if isinstance(obj, wavenumber_exit_time):
            return obj.k.store_id

        raise RuntimeError(
            f'Could not determine shard index k for object of type "{type(obj)}"'
        )

    def _assign_shard_keys(self, obj):
        if isinstance(obj, list):
            data = obj
        else:
            data = [obj]

        # assign any shard keys that we can, without going out to the database
        # (because this is bound to be slower)
        missing_keys = []
        for item in data:
            if not isinstance(item, wavenumber):
                raise RuntimeError("shard keys should be of type wavenumber")

            if item.store_id not in self._wavenumber_keys:
                missing_keys.append(item)

        # if no work to do, return
        if len(missing_keys) == 0:
            return

        # otherwise, we have to populate keys
        # try to load balance by working out which shard has fewest wavenumbers
        loads = {key: 0 for key in self._shards.keys()}
        for shard in self._wavenumber_keys.values():
            loads[shard] = loads[shard] + 1

        with self._engine.begin() as conn:
            for item in missing_keys:
                # find which shard has the current minimum load
                if len(loads) > 0:
                    new_shard = min(loads, key=loads.get)
                else:
                    new_shard = list(self._shards.keys()).pop()

                # insert a new record for this key
                conn.execute(
                    sqla.insert(self._shard_key_table),
                    {"wavenumber_id": item.store_id, "shard_id": new_shard},
                )

                self._wavenumber_keys[item.store_id] = new_shard
                loads[new_shard] = loads[new_shard] + 1

                # print(
                #     f">> assigned shard #{new_shard} to wavenumber object #{item.store_id} (k={item.k_inv_Mpc}/Mpc)"
                # )

            conn.commit()

    def read_wavenumber_table(self, units: UnitsLike):
        """
        Read the wavenumber value table from one of the database shards
        :param units:
        :return:
        """
        # we only need to read the wavenumber table from a single shard, so pick one at random
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        return self._shards[shard_key].read_wavenumber_table.remote(units=units)

    def read_redshift_table(self):
        """
        Read the redshift value table from one of the database shards
        :return:
        """
        # we only need to read the redshift table from a single shard, so pick one at random
        shard_ids = list(self._shards.keys())
        i = random.randrange(len(shard_ids))

        # swap this entry with the last element, then pop it
        shard_ids[i], shard_ids[-1] = shard_ids[-1], shard_ids[i]
        shard_key = shard_ids.pop()

        return self._shards[shard_key].read_redshift_table.remote()
