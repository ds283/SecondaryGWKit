"""
Datastore factories for BackgroundModel and its per-redshift BackgroundModelValue rows.

SCHEMA NOTE (prompts/GkTk-remedial, prompts 03 and 04). Three background primitives are held as
Gauss-Legendre node tables rather than integrated as ODEs, and four columns of
BackgroundModelValue carry them:

  * "tau_Mpc" / "tau_lo_Mpc"       -- the high and low limbs of the double-double conformal time
  * "cs_tau_Mpc" / "cs_tau_lo_Mpc" -- the two limbs of the sound horizon int c_s dz/H
  * "friction_F"                   -- the Liouville-Green friction integral, dimensionless and
                                      stored without a unit conversion; a single double suffices
                                      (|F| <= 60, and it enters as exp(F - F_i))

A datastore whose BackgroundModelValue table lacks any of them predates these changes and must be
regenerated; there is no migration. Its "tau_Mpc" values would in any case be the RK45 solution
the table replaced, not the high limb of the Gauss-Legendre table, so reading them back would
silently restore the 1.4e-9 relative error the change removes, and it has no sound-horizon or
friction samples at all. build() below raises with that message rather than failing on the
missing column.

SCHEMA NOTE (prompts/qcd-background-audit, prompt 14). Two further columns, "source_grid_digest"
and "source_grid_construction", carry the identity of the source sample grid the model was
tabulated on: a content digest of the grid's exact values (prompt 11) and the version of the
algorithm that built it (CosmologyConcepts.wavenumber.SOURCE_GRID_CONSTRUCTION_VERSION).

Without them this factory keyed on (cosmology_type, cosmology_serial, atol_serial, rtol_serial) --
the tolerance half of which prompt 05 of prompts/tolerance-convergence has since replaced with the
three Gauss orders, see the last note below -- plus whatever tags the caller supplied: in main.py,
LargestSourceZTag, SmallestSourceZTag and
(until prompts/qcd-background-audit prompt 16 retired it) SourceSamplesPerLog10ZTag. Every one of
those is unchanged when the grid's *shape* changes, and the factory never filters on z_sample at
all: it reads the stored sample set back out of
BackgroundModelValue and populates the returned object from it. So when prompt 11 gave the QCD
grid 41 extra samples around the equation of state's crossings, a pre-prompt-11 datastore went on
serving its 1,732-node background for the new 1,773-node grid, and the next run tabulated a
1,773-sample pipeline against it. That was the one surviving row in a store whose every compute
target the grid-tag change had already invalidated.

The two columns are used differently on the two paths, and the difference is deliberate:

  * the **compute path** (main.py) supplies z_sample, so build() filters on both -- a model built
    on a different grid, or by a different construction, must miss. A datastore whose
    BackgroundModel table lacks the columns predates prompt 14 and cannot be written to: there is
    no defensible grid identity to assume for its rows, so build() raises with that message
    rather than letting a SQLAlchemy error escape, exactly as sqla_QCDCosmology_factory does.

  * the **read path** (extract_*.py) supplies z_sample=None -- it is asking the store which grid
    was used, not asserting one -- so build() cannot filter on the digest. It selects the two
    columns instead and refuses, naming every generation it found, if the tags it was given
    match rows from more than one. A table without the columns is *not* an error there: its rows
    are reported as an unknown generation and remain readable, because a superseded datastore
    keeps its archival value even once it can no longer serve as a numerical base.

SCHEMA NOTE (prompts/tolerance-convergence, prompt 05). "atol_serial" and "rtol_serial" are gone
and three integer columns -- "tau_gauss_order", "cs_tau_gauss_order", "friction_F_gauss_order" --
stand where they stood in the lookup key.

The pair described nothing. compute_background integrates no ODE: all three primitives are
Gauss-Legendre cumulative tables, so the tolerances reached no solver and were kept only because
they were in the key (ComputeTargets/BackgroundModel.py's own comment said so). What does set the
accuracy is the Gauss order of each table, and those were module constants in no column at all, so
raising TAU_GAUSS_ORDER and re-running left the key unmoved: the pipeline found the existing
order-4 row and served it, while the joined solver_label beside it read "cumulative-GL-stepping4"
-- the truth, recorded, in a column no lookup consulted. Prompt 04 measured all three orders
(docs/tolerance-convergence/ORDER-AUDIT.md) and the user settled the replacement on 2026-09-18
(prompts/tolerance-convergence/README.md §7 D3).

Three columns and not one, because the three primitives are three independent integrals -- 1/H,
c_s/H and -(3/2)(1 + c_s^2)/(1+z) -- each with its own constant and each independently movable.

There is no migration and no default: a BackgroundModel table without the columns is a store from
before this change, and build() raises naming it rather than letting a SQLAlchemy error escape.

SCHEMA NOTE (prompts/tolerance-convergence, prompt 05b). No column changes here, but what fills
the three does. Prompt 05 wrote the value of each module constant, read through an accessor that
re-read it on every call, and build() passed none of the three columns it selected to the
constructor -- so a rehydrated model reassembled its cumulative tables at whatever the module
said, over nodes integrated at whatever the row said. Since 05b build() hands the row's three
orders to the constructor and the compute path echoes the orders compute_background reports, so
an object states the orders its tables carry on both paths. build() still *filters* on the
current module constants (README §7 D10).
"""

from importlib import import_module
from math import fabs
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

# the module, not the class of the same name: ComputeTargets/__init__.py rebinds the attribute
# "BackgroundModel" on the package to the class, so "import ComputeTargets.BackgroundModel as ..."
# would hand back the class. The module is wanted because the three Gauss orders below must be
# resolved at call time from their single declaration, never snapshotted by a from-import.
background_model = import_module("ComputeTargets.BackgroundModel")

from ComputeTargets import (
    BackgroundModel,
    BackgroundModelValue,
)
from CosmologyConcepts import redshift_array, redshift
from CosmologyConcepts.wavenumber import SOURCE_GRID_CONSTRUCTION_VERSION
from CosmologyModels import BaseCosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag
from Quadrature.integration_metadata import IntegrationData, IntegrationSolver
from Units.base import UnitsLike
from config.defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


class sqla_BackgroundModelTagAssociation_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "serial": False,
            "version": False,
            "stepping": False,
            "timestamp": True,
            "columns": [
                sqla.Column(
                    "model_serial",
                    sqla.Integer,
                    sqla.ForeignKey("BackgroundModel.serial"),
                    index=True,
                    nullable=False,
                    primary_key=True,
                ),
                sqla.Column(
                    "tag_serial",
                    sqla.Integer,
                    sqla.ForeignKey("store_tag.serial"),
                    index=True,
                    nullable=False,
                    primary_key=True,
                ),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        raise NotImplementedError

    @staticmethod
    def add_tag(conn, inserter, model: BackgroundModel, tag: store_tag):
        inserter(
            conn,
            {
                "model_serial": model.store_id,
                "tag_serial": tag.store_id,
            },
        )

    @staticmethod
    def remove_tag(conn, table, model: BackgroundModel, tag: store_tag):
        conn.execute(
            sqla.delete(table).where(
                and_(
                    table.c.model_serial == model.store_id,
                    table.c.tag_serial == tag.store_id,
                )
            )
        )


class sqla_BackgroundModelFactory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": True,
            "stepping": False,
            "timestamp": True,
            "validate_on_startup": True,
            "columns": [
                sqla.Column("label", sqla.String(DEFAULT_STRING_LENGTH), nullable=True),
                sqla.Column("cosmology_type", sqla.Integer, index=True, nullable=False),
                sqla.Column(
                    "cosmology_serial", sqla.Integer, index=True, nullable=False
                ),
                # the Gauss-Legendre order of each of the three cumulative tables (module
                # docstring above). These are the accuracy parameters of this object: integers,
                # not tolerances, and each independently movable.
                sqla.Column(
                    "tau_gauss_order", sqla.Integer, index=True, nullable=False
                ),
                sqla.Column(
                    "cs_tau_gauss_order", sqla.Integer, index=True, nullable=False
                ),
                sqla.Column(
                    "friction_F_gauss_order", sqla.Integer, index=True, nullable=False
                ),
                sqla.Column(
                    "solver_serial",
                    sqla.Integer,
                    sqla.ForeignKey("IntegrationSolver.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_init_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column("z_samples", sqla.Integer, nullable=False),
                # the identity of the source grid this model was tabulated on (module docstring
                # above). The digest is a short hex string compared for equality, and the
                # construction version an integer identifier: neither is a measured quantity and
                # DEFAULT_FLOAT_PRECISION has no business near either of them.
                sqla.Column(
                    "source_grid_digest",
                    sqla.String(DEFAULT_STRING_LENGTH),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "source_grid_construction", sqla.Integer, index=True, nullable=False
                ),
                sqla.Column("compute_time", sqla.Float(64)),
                sqla.Column("compute_steps", sqla.Integer),
                sqla.Column("RHS_evaluations", sqla.Integer),
                sqla.Column("mean_RHS_time", sqla.Float(64)),
                sqla.Column("max_RHS_time", sqla.Float(64)),
                sqla.Column("min_RHS_time", sqla.Float(64)),
                sqla.Column("validated", sqla.Boolean, default=False, nullable=False),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        label: Optional[str] = payload.get("label", None)
        tags: List[store_tag] = payload.get("tags", [])

        solver_labels = payload["solver_labels"]

        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_init: redshift_array = payload.get("z_init", None)

        solver_table = tables["IntegrationSolver"]
        tag_table = tables["BackgroundModel_tags"]
        redshift_table = tables["redshift"]

        # the identity of the grid this call is about (module docstring above). The construction
        # version is read from its single declaration, never written out as a literal here: a
        # literal would stop tracking the constant the moment a later prompt bumps it, which is
        # the exact failure this key exists to prevent.
        source_grid_construction = SOURCE_GRID_CONSTRUCTION_VERSION
        source_grid_digest = z_sample.digest() if z_sample is not None else None

        def _build_query(with_grid_identity: bool):
            """
            The lookup. ``with_grid_identity`` selects -- and, on the compute path, filters on --
            the two prompt-14 columns; it is False only for the fallback a pre-prompt-14 table
            forces on the read path, where their absence is archival and not an error.
            """
            columns = [
                table.c.serial,
                table.c.compute_time,
                table.c.compute_steps,
                table.c.RHS_evaluations,
                table.c.mean_RHS_time,
                table.c.max_RHS_time,
                table.c.min_RHS_time,
                table.c.solver_serial,
                table.c.label,
                table.c.z_samples,
                # selected, not merely filtered on: a rehydrated model must report the orders its
                # own row records and rebuild its tables at them
                # (prompts/tolerance-convergence, prompt 05b)
                table.c.tau_gauss_order,
                table.c.cs_tau_gauss_order,
                table.c.friction_F_gauss_order,
                solver_table.c.label.label("solver_label"),
                solver_table.c.stepping.label("solver_stepping"),
            ]
            if with_grid_identity:
                columns.extend(
                    [table.c.source_grid_digest, table.c.source_grid_construction]
                )

            # notice that we query only for validated data
            q = (
                sqla.select(*columns)
                .select_from(
                    table.join(
                        solver_table, solver_table.c.serial == table.c.solver_serial
                    )
                )
                .filter(
                    table.c.validated == True,
                    table.c.cosmology_type == cosmology.type_id,
                    table.c.cosmology_serial == cosmology.store_id,
                    # the accuracy half of the key. Read from the single declaration of each
                    # order at call time, never inlined as a literal: a literal would stop
                    # tracking the constant the moment a later prompt moved it, which is the
                    # exact failure this key exists to prevent
                    table.c.tau_gauss_order == background_model.TAU_GAUSS_ORDER,
                    table.c.cs_tau_gauss_order == background_model.CS_TAU_GAUSS_ORDER,
                    table.c.friction_F_gauss_order
                    == background_model.FRICTION_F_GAUSS_ORDER,
                )
            )

            if with_grid_identity and source_grid_digest is not None:
                # the compute path: a model tabulated on a different grid, or built by a
                # different construction, is a different background and must miss
                q = q.filter(
                    table.c.source_grid_digest == source_grid_digest,
                    table.c.source_grid_construction == source_grid_construction,
                )

            if z_init is not None:
                q = q.filter(
                    table.c.z_init_serial == z_init.store_id,
                )

            # require that the integration we search for has the specified list of tags
            count = 0
            for tag in tags:
                tag: store_tag
                tab = tag_table.alias(f"tag_{count}")
                count += 1
                q = q.join(
                    tab,
                    and_(
                        tab.c.model_serial == table.c.serial,
                        tab.c.tag_serial == tag.store_id,
                    ),
                )

            return q

        # the generation of the row that was served, as (construction version, digest), or None
        # if this datastore predates the columns that record it
        grid_identity = None

        try:
            rows = list(conn.execute(_build_query(True)))
        except SQLAlchemyError as e:
            # matched against the driver's own message rather than str(e), which also carries
            # the SQL statement -- and the statement names every column the query asks for,
            # present or missing, so the three order columns appear in it either way
            detail = str(getattr(e, "orig", e))
            missing_orders = [
                column
                for column in (
                    "tau_gauss_order",
                    "cs_tau_gauss_order",
                    "friction_F_gauss_order",
                )
                if column in detail
            ]
            if len(missing_orders) > 0:
                # a store from before prompts/tolerance-convergence prompt 05. Its rows were
                # keyed on a tolerance pair that reached no solver and record no Gauss order at
                # all, so there is no order that may be assumed for them on either path.
                raise RuntimeError(
                    "BackgroundModel.build(): the BackgroundModel table has no "
                    f'"{missing_orders[0]}" column. This datastore predates the Gauss orders '
                    "replacing the vestigial atol/rtol pair in the background model's lookup key "
                    "(prompts/tolerance-convergence, prompt 05) and must be regenerated; there is "
                    "no migration. Its rows record no order, and the tolerances they were keyed "
                    "on reached no solver, so there is nothing to infer one from."
                ) from e

            if not any(
                column in str(e)
                for column in ("source_grid_digest", "source_grid_construction")
            ):
                raise

            if z_sample is not None:
                # the compute path. There is no defensible grid identity to assume for a row that
                # records none, so this fails loudly rather than silently matching -- the defect
                # being repaired is precisely a stale row that looked like a hit.
                raise RuntimeError(
                    "BackgroundModel.build(): the BackgroundModel table has no "
                    '"source_grid_digest" / "source_grid_construction" columns. This datastore '
                    "predates the source grid becoming part of the background model's lookup key "
                    "(prompts/qcd-background-audit, prompt 14) and must be regenerated; there is "
                    "no migration. Its rows record no grid identity, so a background tabulated on "
                    "a grid that no longer exists cannot be told apart from one tabulated on the "
                    "grid this run builds. Read such a store with the extract_*.py scripts, which "
                    "report it as an unknown generation rather than failing."
                ) from e

            # the read path. A superseded datastore keeps its archival value: report the rows as
            # an unknown generation rather than making them unreadable.
            rows = list(conn.execute(_build_query(False)))

        if len(rows) > 1:
            generations = sorted(
                {
                    (row.source_grid_construction, row.source_grid_digest)
                    for row in rows
                    if hasattr(row, "source_grid_digest")
                }
            )
            if len(generations) == 0:
                found = "an unknown generation (this datastore records none)"
            else:
                found = ", ".join(
                    f"construction version {construction}, grid digest {digest}"
                    for construction, digest in generations
                )
            # "spanning N generations" is the case this exists for -- one label reused across a
            # changed configuration -- but two rows of the *same* generation are no more
            # distinguishable, so both refuse and both name what was found
            raise RuntimeError(
                f"BackgroundModel.build(): the tags supplied match {len(rows)} background models, "
                f"spanning {len(generations)} source grid generation(s) ({found}). A query that "
                "cannot tell them apart must refuse rather than pick one. Narrow the selection -- "
                "for the extract_*.py scripts, with --run-label -- or regenerate the superseded "
                "generation."
            )

        row_data = rows[0] if len(rows) == 1 else None

        if row_data is not None and hasattr(row_data, "source_grid_digest"):
            grid_identity = (
                row_data.source_grid_construction,
                row_data.source_grid_digest,
            )

        if row_data is None:
            # build and return an unpopulated object
            obj = BackgroundModel(
                payload=None,
                solver_labels=solver_labels,
                cosmology=cosmology,
                z_sample=z_sample,
                label=label,
                tags=tags,
            )
            obj._source_grid_identity = None
            return obj

        store_id = row_data.serial
        store_label = row_data.label

        num_expected_samples = row_data.z_samples

        # read out sample values associated with this integration
        value_table = tables["BackgroundModelValue"]

        sample_query = (
            sqla.select(
                value_table.c.serial,
                value_table.c.z_serial,
                redshift_table.c.z,
                redshift_table.c.source.label("z_is_source"),
                redshift_table.c.response.label("z_is_response"),
                value_table.c.Hubble_GeV,
                value_table.c.wBackground,
                value_table.c.wPerturbations,
                value_table.c.rho_GeV,
                value_table.c.tau_Mpc,
                value_table.c.tau_lo_Mpc,
                value_table.c.cs_tau_Mpc,
                value_table.c.cs_tau_lo_Mpc,
                value_table.c.friction_F,
                value_table.c.T_photon_GeV,
                # don't need to read redundant value of T_photon_Kelvin
                value_table.c.d_lnH_dz,
                value_table.c.d2_lnH_dz2,
                value_table.c.d3_lnH_dz3,
                value_table.c.d_wPerturbations_dz,
                value_table.c.d2_wPerturbations_dz2,
            )
            .select_from(
                value_table.join(
                    redshift_table,
                    redshift_table.c.serial == value_table.c.z_serial,
                )
            )
            .filter(value_table.c.model_serial == store_id)
            .order_by(redshift_table.c.z.desc())
        )

        try:
            sample_rows = list(conn.execute(sample_query))
        except SQLAlchemyError as e:
            missing = [
                column
                for column in (
                    "tau_lo_Mpc",
                    "cs_tau_Mpc",
                    "cs_tau_lo_Mpc",
                    "friction_F",
                )
                if column in str(e)
            ]
            if len(missing) > 0:
                raise RuntimeError(
                    "BackgroundModel.build(): the BackgroundModelValue table has no "
                    f'"{missing[0]}" column. This datastore predates the double-double '
                    "conformal-time, sound-horizon and friction node tables "
                    "(prompts/GkTk-remedial, prompts 03 and 04) and must be regenerated; there "
                    "is no migration."
                ) from e
            raise

        z_points = []
        values = []

        units: UnitsLike = cosmology.units
        GeV = units.GeV
        GeV4 = pow(GeV, 4.0)
        Mpc = units.Mpc

        for row in sample_rows:
            z_value = redshift(
                store_id=row.z_serial,
                z=row.z,
                is_source=row.z_is_source,
                is_response=row.z_is_response,
            )
            z_points.append(z_value)
            values.append(
                BackgroundModelValue(
                    store_id=row.serial,
                    z=z_value,
                    Hubble=row.Hubble_GeV * GeV,
                    wBackground=row.wBackground,
                    wPerturbations=row.wPerturbations,
                    rho=row.rho_GeV * GeV4,
                    tau=row.tau_Mpc * Mpc,
                    T_photon=row.T_photon_GeV * GeV,
                    d_lnH_dz=row.d_lnH_dz,
                    d2_lnH_dz2=row.d2_lnH_dz2,
                    d3_lnH_dz3=row.d3_lnH_dz3,
                    d_wPerturbations_dz=row.d_wPerturbations_dz,
                    d2_wPerturbations_dz2=row.d2_wPerturbations_dz2,
                    tau_lo=row.tau_lo_Mpc * Mpc,
                    cs_tau=row.cs_tau_Mpc * Mpc,
                    cs_tau_lo=row.cs_tau_lo_Mpc * Mpc,
                    # dimensionless: no unit conversion on the way in or out
                    friction_F=row.friction_F,
                )
            )
        imported_z_sample = redshift_array(z_points)

        if num_expected_samples is not None:
            if len(imported_z_sample) != num_expected_samples:
                raise RuntimeError(
                    f'Fewer z-samples than expected were recovered from the validated background model "{store_label}"'
                )

        obj = BackgroundModel(
            payload={
                "store_id": store_id,
                "data": IntegrationData(
                    compute_time=row_data.compute_time,
                    compute_steps=row_data.compute_steps,
                    RHS_evaluations=row_data.RHS_evaluations,
                    mean_RHS_time=row_data.mean_RHS_time,
                    max_RHS_time=row_data.max_RHS_time,
                    min_RHS_time=row_data.min_RHS_time,
                ),
                "solver": (
                    IntegrationSolver(
                        store_id=row_data.solver_serial,
                        label=(row_data.solver_label),
                        stepping=(row_data.solver_stepping),
                    )
                ),
                "values": values,
                # the row's own orders, which are the orders this model's three tables were
                # tabulated at. build() filters on the current module constants, so these are
                # those constants today; they are read off the row all the same, because what an
                # object reports -- and what _build_*_primitive reassembles its tables at -- must
                # be a property of the object and not of the module
                # (prompts/tolerance-convergence, prompt 05b)
                "tau_gauss_order": row_data.tau_gauss_order,
                "cs_tau_gauss_order": row_data.cs_tau_gauss_order,
                "friction_F_gauss_order": row_data.friction_F_gauss_order,
            },
            solver_labels=solver_labels,
            cosmology=cosmology,
            z_sample=imported_z_sample,
            label=store_label,
            tags=tags,
        )
        obj._deserialized = True
        # the generation this row belongs to, for the read path to report: (construction version,
        # digest), or None for a datastore that predates prompt 14 and records neither
        obj._source_grid_identity = grid_identity
        return obj

    @staticmethod
    def store(
        obj: BackgroundModel,
        conn,
        table,
        inserter,
        tables,
        inserters,
    ):
        payload = {
            "label": obj.label,
            "cosmology_type": obj.cosmology.type_id,
            "cosmology_serial": obj.cosmology.store_id,
            # the three orders this model's tables were actually built at: since prompt 05b the
            # accessors report the object's own orders -- echoed out of the compute_background
            # payload on the compute path, read off the row on the rehydration path -- rather
            # than re-reading the module constants. build() goes on filtering on those constants,
            # so a model tabulated at another order is simply a different row (README §7 D10).
            "tau_gauss_order": obj.tau_gauss_order,
            "cs_tau_gauss_order": obj.cs_tau_gauss_order,
            "friction_F_gauss_order": obj.friction_F_gauss_order,
            "solver_serial": obj.solver.store_id,
            "z_init_serial": obj.z_sample.min.store_id,
            "z_samples": len(obj.values),
            # the grid this model was tabulated on, written from the same two sources build()
            # filters on: the digest of the sample set itself, and the single declaration of the
            # construction version. Neither is a literal here, so the stored value and the
            # queried value cannot disagree.
            "source_grid_digest": obj.z_sample.digest(),
            "source_grid_construction": SOURCE_GRID_CONSTRUCTION_VERSION,
            "compute_time": obj.data.compute_time,
            "compute_steps": obj.data.compute_steps,
            "RHS_evaluations": obj.data.RHS_evaluations,
            "mean_RHS_time": obj.data.mean_RHS_time,
            "max_RHS_time": obj.data.max_RHS_time,
            "min_RHS_time": obj.data.min_RHS_time,
            "validated": False,
        }

        # because BackgroundModel is a replicated table, we need to allow for the possibility that this object
        # is a replica, rather than a fresh insert. If so, it's _my_id field will be set.
        if hasattr(obj, "_my_id") and obj._my_id is not None:
            payload.update({"serial": obj._my_id})

        store_id = inserter(conn, payload)

        # set store_id on behalf of the BackgroundModel instance
        obj._my_id = store_id

        # add any tags that have been specified
        tag_inserter = inserters["BackgroundModel_tags"]
        for tag in obj.tags:
            sqla_BackgroundModelTagAssociation_factory.add_tag(
                conn, tag_inserter, obj, tag
            )

        # now serialize the sampled output points
        units: UnitsLike = obj._units
        GeV = units.GeV
        GeV4 = pow(GeV, 4.0)
        Kelvin = units.Kelvin
        Mpc = units.Mpc

        value_inserter = inserters["BackgroundModelValue"]
        for value in obj.values:
            value: BackgroundModelValue
            value_id = value_inserter(
                conn,
                {
                    "model_serial": store_id,
                    "z_serial": value.z.store_id,
                    "Hubble_GeV": value.Hubble / GeV,
                    "rho_GeV": value.rho / GeV4,
                    "wBackground": value.wBackground,
                    "wPerturbations": value.wPerturbations,
                    "tau_Mpc": value.tau / Mpc,
                    "tau_lo_Mpc": value.tau_lo / Mpc,
                    "cs_tau_Mpc": value.cs_tau / Mpc,
                    "cs_tau_lo_Mpc": value.cs_tau_lo / Mpc,
                    "friction_F": value.friction_F,
                    "T_photon_GeV": value.T_photon / GeV,
                    "T_photon_Kelvin": value.T_photon / Kelvin,
                    "d_lnH_dz": value.d_lnH_dz,
                    "d2_lnH_dz2": value.d2_lnH_dz2,
                    "d3_lnH_dz3": value.d3_lnH_dz3,
                    "d_wPerturbations_dz": value.d_wPerturbations_dz,
                    "d2_wPerturbations_dz2": value.d2_wPerturbations_dz2,
                },
            )

            # set store_id on behalf of the GkNumericValue instance
            value._my_id = value_id

        return obj

    @staticmethod
    def validate(
        obj: BackgroundModel,
        conn,
        table,
        tables,
    ):
        # query the row in BackgroundModel corresponding to this object
        if not obj.available:
            raise RuntimeError(
                "Attempt to validate a datastore object that has not yet been serialized"
            )

        expected_samples = conn.execute(
            sqla.select(table.c.z_samples).filter(table.c.serial == obj.store_id)
        ).scalar()

        value_table = tables["BackgroundModelValue"]
        num_samples = conn.execute(
            sqla.select(sqla.func.count(value_table.c.serial)).filter(
                value_table.c.model_serial == obj.store_id
            )
        ).scalar()

        # check if we counted as many rows as we expected
        validated: bool = num_samples == expected_samples
        if not validated:
            print(
                f'!! WARNING: Background model "{obj.label}" did not validate after serialization (expected samples={expected_samples}, number stored={num_samples})'
            )

        conn.execute(
            sqla.update(table)
            .where(table.c.serial == obj.store_id)
            .values(validated=validated)
        )

        return validated

    @staticmethod
    def validate_on_startup(conn, table, tables, prune=False):
        # query the datastore for any integrations that are not validated

        solver_table = tables["IntegrationSolver"]
        value_table = tables["BackgroundModelValue"]
        tags_table = tables["BackgroundModel_tags"]

        # bake results into a list so that we can close this query; we are going to want to run
        # another one as we process the rows from this one
        not_validated = list(
            conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.label,
                    table.c.z_samples,
                    solver_table.c.label.label("solver_label"),
                    table.c.tau_gauss_order,
                    table.c.cs_tau_gauss_order,
                    table.c.friction_F_gauss_order,
                )
                .select_from(
                    table.join(
                        solver_table, solver_table.c.serial == table.c.solver_serial
                    )
                )
                .filter(or_(table.c.validated == False, table.c.validated == None))
            )
        )

        if len(not_validated) == 0:
            return []

        msgs = [
            ">> Background models",
            "     The following unvalidated models were detected in the datastore:",
        ]
        for model in not_validated:
            msgs.append(
                f'       -- "{model.label}" (store_id={model.serial}, '
                f"N_tau={model.tau_gauss_order}, N_cs_tau={model.cs_tau_gauss_order}, "
                f"N_F={model.friction_F_gauss_order})"
            )
            rows = conn.execute(
                sqla.select(sqla.func.count(value_table.c.serial)).filter(
                    value_table.c.model_serial == model.serial,
                )
            ).scalar()
            msgs.append(
                f"          contains {rows} z-sample values | expected={model.z_samples}"
            )

        if prune:
            invalid_serials = [nv.serial for nv in not_validated]
            try:
                conn.execute(
                    sqla.delete(value_table).where(
                        value_table.c.model_serial.in_(invalid_serials)
                    )
                )
                conn.execute(
                    sqla.delete(tags_table).where(
                        tags_table.c.model_serial.in_(invalid_serials)
                    )
                )
                conn.execute(
                    sqla.delete(table).where(table.c.serial.in_(invalid_serials))
                )
            except SQLAlchemyError:
                msgs.append(
                    f"!!        DATABASE ERROR encountered when pruning these values"
                )
                pass
            else:
                msgs.append(
                    f"     ** Note: these values have been pruned from the datastore."
                )

        return msgs

    @staticmethod
    def inventory_records(conn, table, tables, context):
        # the physical identity of a BackgroundModel row (store-fingerprint prompt 02): what build()
        # filters on -- the cosmology (type and row), the three Gauss orders, the source grid's
        # construction and digest, and z_init (optional in the lookup, always in the key) -- with
        # its tags, its validated flag and its BackgroundModelValue count. The solver is not in the
        # lookup and is not identity
        from Datastore.store_inventory import COSMOLOGY, Parent, read_records

        return read_records(
            conn,
            table,
            tables,
            context,
            leaves=(
                "cosmology_type",
                "tau_gauss_order",
                "cs_tau_gauss_order",
                "friction_F_gauss_order",
                "source_grid_construction",
                "source_grid_digest",
            ),
            parents={
                "cosmology": Parent("cosmology_serial", COSMOLOGY, "cosmology_type"),
                "z_init": Parent("z_init_serial", "redshift"),
            },
            tags=("BackgroundModel_tags", "model_serial"),
            values=("BackgroundModelValue", "model_serial"),
            validated=True,
        )


class sqla_BackgroundModelValue_factory(SQLAFactoryBase):
    def __init__(self):
        pass

    @staticmethod
    def register():
        return {
            "version": False,
            "timestamp": False,
            "stepping": False,
            "columns": [
                sqla.Column(
                    "model_serial",
                    sqla.Integer,
                    sqla.ForeignKey("BackgroundModel.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "z_serial",
                    sqla.Integer,
                    sqla.ForeignKey("redshift.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column("Hubble_GeV", sqla.Float(64), nullable=False),
                sqla.Column("wBackground", sqla.Float(64), nullable=False),
                sqla.Column("wPerturbations", sqla.Float(64), nullable=False),
                sqla.Column("rho_GeV", sqla.Float(64), nullable=False),
                sqla.Column("tau_Mpc", sqla.Float(64), nullable=False),
                sqla.Column("tau_lo_Mpc", sqla.Float(64), nullable=False),
                sqla.Column("cs_tau_Mpc", sqla.Float(64), nullable=False),
                sqla.Column("cs_tau_lo_Mpc", sqla.Float(64), nullable=False),
                # dimensionless, so no unit suffix and no conversion
                sqla.Column("friction_F", sqla.Float(64), nullable=False),
                sqla.Column("T_photon_GeV", sqla.Float(64), nullable=False),
                sqla.Column("T_photon_Kelvin", sqla.Float(64), nullable=False),
                sqla.Column("d_lnH_dz", sqla.Float(64), nullable=False),
                sqla.Column("d2_lnH_dz2", sqla.Float(64), nullable=False),
                sqla.Column("d3_lnH_dz3", sqla.Float(64), nullable=False),
                sqla.Column("d_wPerturbations_dz", sqla.Float(64), nullable=False),
                sqla.Column("d2_wPerturbations_dz2", sqla.Float(64), nullable=False),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        model_serial = payload["model_serial"]
        units = payload["units"]

        z = payload["z"]

        Hubble = payload["Hubble"]
        wBackground = payload["wBackground"]
        wPerturbations = payload["wPerturbations"]

        rho = payload["rho"]
        tau = payload["tau"]
        tau_lo = payload["tau_lo"]
        cs_tau = payload["cs_tau"]
        cs_tau_lo = payload["cs_tau_lo"]
        friction_F = payload["friction_F"]
        T_photon = payload["T_photon"]

        d_lnH_dz = payload["d_lnH_dz"]
        d2_lnH_dz2 = payload["d2_lnH_dz2"]
        d3_lnH_dz3 = payload["d3_lnH_dz3"]

        d_wPerturbations_dz = payload["d_wPerturbations_dz"]
        d2_wPerturbations_dz2 = payload["d2_wPerturbations_dz2"]

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.Hubble_GeV,
                    table.c.rho_GeV,
                    table.c.wBackground,
                    table.c.wPerturbations,
                    table.c.tau_Mpc,
                    table.c.tau_lo_Mpc,
                    table.c.cs_tau_Mpc,
                    table.c.cs_tau_lo_Mpc,
                    table.c.friction_F,
                    table.c.T_photon_GeV,
                    # don't need to read redundant value of T_photon_Kelvin
                    table.c.d_lnH_dz,
                    table.c.d2_lnH_dz2,
                    table.c.d3_lnH_dz3,
                    table.c.d_wPerturbations_dz,
                    table.c.d2_wPerturbations_dz2,
                ).filter(
                    table.c.model_serial == model_serial,
                    table.c.z_serial == z.store_id,
                )
            ).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! BackgroundModelValue.build(): multiple results found when querying for BackgroundModelValue"
            )
            raise e

        GeV = units.GeV
        GeV4 = pow(GeV, 4.0)
        Kelvin = units.Kelvin
        Mpc = units.Mpc

        if row_data is None:
            store_id = inserter(
                conn,
                {
                    "model_serial": model_serial,
                    "z_serial": z.store_id,
                    "Hubble_GeV": Hubble / GeV,
                    "rho_GeV": rho / GeV4,
                    "wBackground": wBackground,
                    "wPerturbations": wPerturbations,
                    "tau_Mpc": tau / Mpc,
                    "tau_lo_Mpc": tau_lo / Mpc,
                    "cs_tau_Mpc": cs_tau / Mpc,
                    "cs_tau_lo_Mpc": cs_tau_lo / Mpc,
                    "friction_F": friction_F,
                    "T_photon_GeV": T_photon / GeV,
                    "T_photon_Kelvin": T_photon / Kelvin,
                    "d_lnH_dz": d_lnH_dz,
                    "d2_lnH_dz2": d2_lnH_dz2,
                    "d3_lnH_dz3": d3_lnH_dz3,
                    "d_wPerturbations_dz": d_wPerturbations_dz,
                    "d2_wPerturbations_dz2": d2_wPerturbations_dz2,
                },
            )
        else:
            store_id = row_data.serial

            # the two consistency checks run here, BEFORE the payload values are replaced by the
            # stored ones below. Each asks whether what the caller wants agrees with what is on
            # disk, and after the replacement there is nothing left to compare against: the
            # Hubble check used to sit below and read a "Hubble" that had already become the
            # stored value.
            #
            # Hubble is compared in its stored representation and relatively, because no single
            # absolute bound can serve a column spanning thirty decades. Over a production
            # redshift grid the internal-unit value runs from 2.37e-04 to 9.19e+26: at the bottom
            # DEFAULT_FLOAT_PRECISION is a relative tolerance of 4.2e-04, loose enough to accept a
            # stored Hubble wrong in its fourth significant figure, while at the top one ulp is
            # 1.37e+11, so the smallest disagreement a double can express is 1.4e+18 times the
            # bound and the test degenerates to bit equality. wBackground is dimensionless and
            # O(1), so the absolute bound is the right instrument for it and is unchanged.
            expected_Hubble_GeV = Hubble / GeV
            if fabs(
                row_data.Hubble_GeV - expected_Hubble_GeV
            ) > DEFAULT_FLOAT_PRECISION * fabs(expected_Hubble_GeV):
                raise ValueError(
                    f"Stored background model Hubble value (model store_id={model_serial}, z={z.store_id}) = {row_data.Hubble_GeV} GeV differs from expected value = {expected_Hubble_GeV} GeV"
                )
            if fabs(row_data.wBackground - wBackground) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored background model w_Background value (model store_id={model_serial}, z={z.store_id}) = {row_data.wBackground} differs from expected value = {wBackground}"
                )

            Hubble = row_data.Hubble_GeV * GeV
            rho = row_data.rho_GeV * GeV4
            tau = row_data.tau_Mpc * Mpc
            tau_lo = row_data.tau_lo_Mpc * Mpc
            cs_tau = row_data.cs_tau_Mpc * Mpc
            cs_tau_lo = row_data.cs_tau_lo_Mpc * Mpc
            friction_F = row_data.friction_F
            T_photon = row_data.T_photon_GeV * GeV

            d_lnH_dz = row_data.d_lnH_dz
            d2_lnH_dz2 = row_data.d2_lnH_dz2
            d3_lnH_dz3 = row_data.d3_lnH_dz3

            d_wPerturbations_dz = row_data.d_wPerturbations_dz
            d2_wPerturbations_dz2 = row_data.d2_wPerturbations_dz2

        obj = BackgroundModelValue(
            store_id=store_id,
            z=z,
            Hubble=Hubble,
            rho=rho,
            wBackground=wBackground,
            wPerturbations=wPerturbations,
            tau=tau,
            T_photon=T_photon,
            d_lnH_dz=d_lnH_dz,
            d2_lnH_dz2=d2_lnH_dz2,
            d3_lnH_dz3=d3_lnH_dz3,
            d_wPerturbations_dz=d_wPerturbations_dz,
            d2_wPerturbations_dz2=d2_wPerturbations_dz2,
            tau_lo=tau_lo,
            cs_tau=cs_tau,
            cs_tau_lo=cs_tau_lo,
            friction_F=friction_F,
        )
        obj._deserialized = True
        return obj
