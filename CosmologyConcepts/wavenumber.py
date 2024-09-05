from typing import Iterable

import ray
from ray.actor import ActorHandle

import sqlalchemy as sqla
from sqlalchemy import func, and_, literal_column

from CosmologyConcepts.tolerance import tolerance
from CosmologyModels.base import CosmologyBase
from Datastore import DatastoreObject
from defaults import DEFAULT_FLOAT_PRECISION
from utilities import find_horizon_exit_time, check_units, WallclockTimer


class wavenumber(DatastoreObject):
    def __init__(self, store: ActorHandle, k_inv_Mpc: float, units):
        """
        Construct a datastore-backed object representing a wavenumber, e.g.,
        used to sample a transfer function or power spectrum
        :param store: handle to datastore actor
        :param k_inv_Mpc: wavenumber, measured in 1/Mpc
        :param units: units block (e.g. Mpc-based units)
        """
        DatastoreObject.__init__(self, store)

        # units are available for inspection
        self.units = units

        self.k_inv_Mpc = k_inv_Mpc
        self.k = k_inv_Mpc / units.Mpc

        # request our own unique id from the datastore
        self._my_id = ray.get(self._store.query.remote(self))

    def __float__(self):
        """
        Cast to float. Returns dimensionful wavenumber.
        :return:
        """
        return self.k

    @staticmethod
    def generate_columns():
        return {"version": False, "timestamp": True, "columns": [sqla.Column("k_inv_Mpc", sqla.Float(64))]}

    def build_query(self, table, query):
        return query.filter(
            func.abs(table.c.k_inv_Mpc - self.k_inv_Mpc) < DEFAULT_FLOAT_PRECISION
        )

    def build_storage_payload(self):
        return {"k_inv_Mpc": self.k_inv_Mpc}


class wavenumber_array:
    def __init__(self, store: ActorHandle, k_inv_Mpc_array: Iterable, units):
        """
        Construct a datastore-backed object representing an array of wavenumber values
        :param store: handle to datastore actor
        :param k_inv_Mpc_array: array of wavenumbers, measured in 1/Mpc
        :param units: units block
        """
        self._store: ActorHandle = store

        # build array
        self._k_array = [
            wavenumber(store, k_inv_Mpc, units) for k_inv_Mpc in k_inv_Mpc_array
        ]

    def __iter__(self):
        for k in self._k_array:
            yield k

    def __getitem__(self, key):
        return self._k_array[key]


class wavenumber_exit_time(DatastoreObject):
    def __init__(
        self,
        store: ActorHandle,
        k: wavenumber,
        cosmology: CosmologyBase,
        target_atol: tolerance = None,
        target_rtol: tolerance = None,
    ):
        """
        Construct a datastore-backed object representing the horizon exit time
        for a mode of wavenumber k
        :param store: handle to datastore actor
        :param k: wavenumber object
        :param cosmology: cosmology object satisfying the CosmologyBase concept
        """
        DatastoreObject.__init__(self, store)
        check_units(k, cosmology)

        self.k = k
        self.cosmology = cosmology

        self._z_exit = None

        # obtain and cache handle to table of tolerance values
        # also, set up aliases for seprate atol and rtol columns
        self._tolerance_table: sqla.Table = ray.get(store.table.remote(tolerance))
        self._atol_table = self._tolerance_table.alias("atol_table")
        self._rtol_table = self._tolerance_table.alias("rtol_table")

        # cache requested tolerances
        if target_atol is None:
            target_atol = tolerance(store, tol=1e-5)
        if target_rtol is None:
            target_rtol = tolerance(store, tol=1e-7)

        self._target_atol: tolerance = target_atol
        self._target_rtol: tolerance = target_rtol

        # request our own unique id from the datastore
        db_info = ray.get(self._store.query.remote(self, serial_only=False))

        self._my_id = db_info["store_id"]
        row = db_info["data"]

        self._z_exit = row["z_exit"]
        self._atol: tolerance = tolerance(store, log10_tol=row["log10_atol"])
        self._rtol: tolerance = tolerance(store, log10_tol=row["log10_rtol"])

    @staticmethod
    def generate_columns():
        # Does not set up a foreign key constraint for the cosmology object.
        # The problem is that this is polymorphic, because we have different implementations of the CosmologyBase concept.
        # Rather than try to deal with this using SQLAlchemy-level polymorphism, we handle the polymorphism ourselves
        # and just skip foreign key constraints here
        return {
            "version": True,
            "timestamp": True,
            "stepping": "minimum",
            "columns": [
                sqla.Column(
                    "wavenumber_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber.serial"),
                    nullable=False,
                ),
                sqla.Column("cosmology_type", sqla.Integer, nullable=False),
                sqla.Column("cosmology_serial", sqla.Integer, nullable=False),
                sqla.Column(
                    "atol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "rtol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    nullable=False,
                ),
                sqla.Column("time", sqla.Float(64)),
                sqla.Column("z_exit", sqla.Float(64)),
            ],
        }

    @property
    def stepping(self):
        # stepping 0: initial implementation using solve_ivp and an event handler to determine when ln(k/aH) = 0
        return 0

    def build_query(self, table, query):
        # query for an existing record that at least matches the specified tolerances
        # notice we have to replace the .select_from() specifier that has been pre-populated by the DataStore object
        # If we just have .select_from(BASE_TABLE), we cannot access any columns from the joined tables (at least using SQLite)
        # see: https://stackoverflow.com/questions/68137220/getting-columns-of-the-joined-table-using-sqlalchemy-core

        # order by descending values of abs and relative tolerances, so that we get the best computed value we hold
        query = (
            sqla.select(
                table.c.serial,
                table.c.version,
                table.c.stepping,
                table.c.timestamp,
                table.c.wavenumber_serial,
                table.c.cosmology_type,
                table.c.cosmology_serial,
                table.c.atol_serial,
                table.c.rtol_serial,
                table.c.time,
                self._atol_table.c.log10_tol.label("log10_atol"),
                self._rtol_table.c.log10_tol.label("log10_rtol"),
                table.c.z_exit,
            )
            .select_from(
                table.join(
                    self._atol_table, self._atol_table.c.serial == table.c.atol_serial
                ).join(
                    self._rtol_table, self._rtol_table.c.serial == table.c.rtol_serial
                )
            )
            .filter(
                and_(
                    table.c.wavenumber_serial == self.k.store_id,
                    table.c.cosmology_type == self.cosmology.type_id,
                    table.c.cosmology_serial == self.cosmology.store_id,
                    self._atol_table.c.log10_tol - self._target_atol.log10_tol
                    <= DEFAULT_FLOAT_PRECISION,
                    self._rtol_table.c.log10_tol - self._target_rtol.log10_tol
                    <= DEFAULT_FLOAT_PRECISION,
                )
            )
            .order_by(self._atol_table.c.log10_tol.desc())
            .order_by(self._rtol_table.c.log10_tol.desc())
        )
        return query

    def build_storage_payload(self):
        with WallclockTimer() as timer:
            self._z_exit = find_horizon_exit_time(
                self.cosmology,
                self.k,
                atol=self._target_atol.tol,
                rtol=self._target_rtol.tol,
            )

        return {
            "wavenumber_serial": self.k.store_id,
            "cosmology_type": self.cosmology.type_id,
            "cosmology_serial": self.cosmology.store_id,
            "atol_serial": self._target_atol.store_id,
            "rtol_serial": self._target_rtol.store_id,
            "log10_atol": self._target_atol.log10_tol,
            "log10_rtol": self._target_rtol.log10_tol,
            "time": timer.elapsed,
            "z_exit": self._z_exit,
        }

    @property
    def z_exit(self) -> float:
        return self._z_exit

    @property
    def atol(self) -> float:
        return self._atol.tol

    @property
    def rtol(self) -> float:
        return self._rtol.tol
