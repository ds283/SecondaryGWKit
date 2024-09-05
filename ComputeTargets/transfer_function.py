from math import log10, exp
from typing import Optional

import numpy as np

import ray
import sqlalchemy as sqla
from sqlalchemy import func, and_

from ray.actor import ActorHandle

from ComputeTargets import IntegrationSolver
from CosmologyConcepts import redshift_array, wavenumber, redshift, tolerance
from CosmologyConcepts.wavenumber import wavenumber_exit_time
from CosmologyModels.base import CosmologyBase
from Datastore import DatastoreObject
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class MatterTransferFunctionIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced during a single integration of the
    matter transfer function, labelled by a wavenumber k, and sampled over
    a range of redshifts
    """

    def __init__(
            self,
            store: ActorHandle,
            cosmology: CosmologyBase,
            label: str,
            k: wavenumber,
            z_samples: redshift_array,
            z_init: redshift,
            atol: float,
            rtol: float,
    ):
        DatastoreObject.__init__(self, store)
        self._cosmology = cosmology

        # cache wavenumber and z-sample array
        self._label = label
        self._k = k
        self._z_samples = z_samples
        self._z_init = z_init

        self._atol = tolerance(store, tol=atol)
        self._rtol = tolerance(store, tol=rtol)

        # build label for our solver/strategy
        self._solver = IntegrationSolver(store, label="solve_ivp+RK45", stepping=0)

        # check that all sample points are later than the specified initial redshift
        z_init_float = float(z_init)
        for z in z_samples:
            z_float = float(z)
            if z_float > z_init_float:
                raise RuntimeError(
                    f"Specified sample redshift z={z_float} exceeds initial redshift z={z_init_float}"
                )

        # request our own unique id from the datastore
        self._my_id = ray.get(self._store.query.remote(self))

    @staticmethod
    def generate_columns():
        return {
            "version": True,
            "stepping": False,
            "timestamp": True,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH)),
                sqla.Column(
                    "wavenumber_serial",
                    sqla.Integer,
                    sqla.ForeignKey("wavenumber.serial"),
                    nullable=False,
                ),
                sqla.Column("cosmology_type", sqla.Integer, nullable=False),
                sqla.Column("ccosmology_serial", sqla.Integer, nullable=False),
                sqla.Column(
                    "atol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "atol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "solver_serial",
                    sqla.Integer,
                    sqla.ForeignKey("solver.serial"),
                    nullable=False,
                ),
                sqla.Column("time", sqla.Float(64)),
                sqla.Column("steps", sqla.Integer),
            ],
        }

    def build_query(self, table, query):
        return query.filter(
            and_(
                table.c.wavenumber_serial == self._k.store_id,
                table.c.cosmology_type == self._cosmology.type_id,
                table.c.cosmology_serial == self._cosmology.store_id,
                table.c.label == self._label,
                table.c.atol_serial == self._atol.store_id,
                table.c.rtol_serial == self._rtol.store_id,
            )
        )

    def build_storage_payload(self):
        return {
            "label": self._label,
            "wavenumber_serial": self._k.store_id,
            "cosmology_type": self._cosmology.type_id,
            "cosmology_serial": self._cosmology.store_id,
            "atol_serial": self._atol.store_id,
            "rtol_serial": self._rtol.store_id,
            "solver_serial": self._solver.store_id,
        }

    @property
    def store(self):
        return self._store

    @property
    def cosmology(self):
        return self._cosmology


class MatterTransferFunctionValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the matter transfer functions, labelled by a wavenumber k
    and a redshift z, and an initial redshift z_initial (at which the initial condition T(z_init) = 1.0 applies)
    """

    def __init__(
            self,
            store: ActorHandle,
            cosmology: CosmologyBase,
            k: wavenumber,
            z_init: redshift,
            z: redshift,
            target_atol: float = 1e-5,
            target_rtol: float = 1e-7,
    ):
        DatastoreObject.__init__(self, store)

        self._cosmology: CosmologyBase = cosmology

        # cache parameters and values
        self._k = k
        self._z = z
        self._z_init = z_init

        # obtain and cache handle to table of tolerance values
        # also, set up aliases for seprate atol and rtol columns
        self._tolerance_table: sqla.Table = ray.get(self._store.table.remote(tolerance))
        self._atol_table = self._tolerance_table.alias("atol_table")
        self._rtol_table = self._tolerance_table.alias("rtol_table")

        # convert requested tolerances to database ids
        self._target_atol: tolerance = tolerance(self._store, tol=target_atol)
        self._target_rtol: tolerance = tolerance(self._store, tol=target_rtol)

        # obtain and cache handle to table of integration records
        self._integration_table: sqla.Table = ray.get(store.table.remote(MatterTransferFunctionIntegration))
        self._solver_table: sqla.Table = ray.get(store.table.remote(IntegrationSolver))

        # query whether this sample value is available in the datastore
        db_info = ray.get(store.query.remote(self, serial_only=False))

        self._my_id = db_info["store_id"]
        row = db_info["data"]

        if db_info is None:
            self._my_id = None
            self._value = None
            return

        self._value = row["value"]

    @property
    def available(self) -> bool:
        return self._my_id is not None

    @property
    def value(self) -> Optional[float]:
        if self._my_id is None:
            return None

        return self._value

    @staticmethod
    def generate_columns():
        return {
            "defer_insert": True,
            "version": False,
            "timestamp": False,
            "stepping": False,
            "columns": [
                sqla.Column(
                    "integration_serial",
                    sqla.Integer,
                    sqla.ForeignKey("MatterTransferFunctionIntegration.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "z_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    nullable=False,
                ),
                sqla.Column(
                    "value",
                    sqla.Float(64),
                )
            ],
        }

    def build_query(self, table, query):
        query = (
            sqla.select(
                table.c.serial,
                table.c.integration_serial,
                table.c.z_serial,
                table.c.value,
                self._atol_table.c.log10_tol.label("log10_atol"),
                self._rtol_table.c.log10_tol.label("log10_rtol"),
                self._solver_table.c.label.label("solver_label"),
                self._solver_table.c.stepping.label("solver_stepping"),
                self._integration_table.c.timestamp,
                self._integration_table.c.version
            )
            .select_from(
                table.join(
                    self._integration_table, self._integration_table.c.serial == table.c.integration_serial
                ).join(
                    self._solver_table, self._solver_table.c.serial == self._integration_table.c.solver_serial,
                ).join(
                    self._atol_table, self._atol_table.c.serial == self._integration_table.c.atol_serial
                ).join(
                    self._rtol_table, self._rtol_table.c.serial == self._integration_table.c.rtol_serial
                )
            )
            .filter(
                and_(
                    table.c.z_serial == self._z.store_id,
                    self._integration_table.c.wavenumber_serial == self._k.store_id,
                    self._integration_table.c.cosmology_type == self._cosmology.type_id,
                    self._integration_table.c.cosmology_serial == self._cosmology.store_id,
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


class MatterTransferFunction(DatastoreObject):
    """
    Encapsulates the time-evolution of the matter transfer function, labelled by a wavenumber k,
    and sampled over a specified range of redshifts
    """

    def __init__(
            self,
            store: ActorHandle,
            cosmology: CosmologyBase,
            k: wavenumber,
            z_init: redshift,
            z_samples: redshift_array,
    ):
        """
        :param store: handle to datastore actor
        :param cosmology: cosmology instance
        :param k: wavenumber object
        :param z_initial: initial redshift of the matter transfer function
        :param z_samples: redshift values at which to sample the matter transfer function
        """
        DatastoreObject.__init__(self, store)
        self._cosmology: CosmologyBase = cosmology

        # cache wavenumber and z-sample array
        self._k = k
        self._z_samples = z_samples
        self._z_initial = z_init

        # query datastore to find out whether the necessary sample values are available,
        # and schedule asynchronous tasks to compute any that are missing.

    @classmethod
    def populate_z_samples(
            cls,
            store: ActorHandle,
            cosmology: CosmologyBase,
            k: wavenumber,
            outside_horizon_efolds: float = 10.0,
            samples_per_log10z: int = 100,
            z_end: float = 0.1,
    ):
        """
        Determine the approximate starting redshift, if we wish to begin the calculation when
        the mode k is a fixed number of e-folds outside the horizon, and returns
        a MatterTransferFunction instance populated with an appropriate z-sample
        array. The array using a fixed number of sample points per log10 interval of redshift.

        The calculation of the initial redshift assumes that horizon crossing occurs during
        :param store: handle to datastore actor
        :param cosmology: cosmology models satisfying the CosmologyBase concept
        :param k: the wavenumber to sample
        :param outside_horizon_efolds: number of superhorizon efolds required in the initial condition
        :param samples_per_log10z: number of points to sample per log10 interval of redshift
        :return: MatterTransferFunction instance
        """
        # find horizon-crossing time for wavenumber z
        exit_time = wavenumber_exit_time(store, k, cosmology)

        # add on any specified number of superhorizon e-folds
        # since 1 + z = a0/a, scaling the initial a by exp(-N) scales 1 + z by exp(+N)
        if outside_horizon_efolds is not None and outside_horizon_efolds > 0.0:
            z_init = (1.0 + exit_time.z_exit) * exp(outside_horizon_efolds) - 1.0

        print(
            f"horizon-crossing time for wavenumber k = {k.k_inv_Mpc}/Mpc is z_exit = {exit_time.z_exit}"
        )
        print(
            f"-- using N={outside_horizon_efolds} of superhorizon evolution, initial time is z_init = {z_init}"
        )

        # now we want to sample to transfer function between z_init and z = 0, using the specified
        # number of redshift sample points
        num_z_samples = int(round(samples_per_log10z * log10(z_init) + 0.5, 0))
        z_samples = redshift_array(
            store,
            np.logspace(log10(z_init), log10(z_end), num=num_z_samples),
        )

        print(
            f"-- using N = {num_z_samples} redshift sample points to represent transfer function"
        )

        return cls(
            store,
            cosmology=cosmology,
            k=k,
            z_init=redshift(store, z_init),
            z_samples=z_samples,
        )
