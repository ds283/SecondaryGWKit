from math import fabs
from typing import Optional, List

import sqlalchemy as sqla
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from ComputeTargets import (
    BackgroundModel,
    BackgroundModelValue,
)
from CosmologyConcepts import redshift_array, redshift
from CosmologyModels import BaseCosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance
from Quadrature.integration_metadata import IntegrationData, IntegrationSolver
from Units.base import UnitsLike
from defaults import DEFAULT_STRING_LENGTH, DEFAULT_FLOAT_PRECISION


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
                sqla.Column(
                    "atol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    index=True,
                    nullable=False,
                ),
                sqla.Column(
                    "rtol_serial",
                    sqla.Integer,
                    sqla.ForeignKey("tolerance.serial"),
                    index=True,
                    nullable=False,
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

        atol: tolerance = payload["atol"]
        rtol: tolerance = payload["rtol"]

        cosmology: BaseCosmology = payload["cosmology"]
        z_sample: redshift_array = payload["z_sample"]
        z_init: redshift_array = payload.get("z_init", None)

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        solver_table = tables["IntegrationSolver"]
        tag_table = tables["BackgroundModel_tags"]
        redshift_table = tables["redshift"]

        # notice that we query only for validated data
        query = (
            sqla.select(
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
                solver_table.c.label.label("solver_label"),
                solver_table.c.stepping.label("solver_stepping"),
                atol_table.c.log10_tol.label("log10_atol"),
                rtol_table.c.log10_tol.label("log10_rtol"),
            )
            .select_from(
                table.join(solver_table, solver_table.c.serial == table.c.solver_serial)
                .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
            )
            .filter(
                table.c.validated == True,
                table.c.cosmology_type == cosmology.type_id,
                table.c.cosmology_serial == cosmology.store_id,
                table.c.atol_serial == atol.store_id,
                table.c.rtol_serial == rtol.store_id,
            )
        )

        if z_init is not None:
            query = query.filter(
                table.c.z_init_serial == z_init.store_id,
            )

        # require that the integration we search for has the specified list of tags
        count = 0
        for tag in tags:
            tag: store_tag
            tab = tag_table.alias(f"tag_{count}")
            count += 1
            query = query.join(
                tab,
                and_(
                    tab.c.model_serial == table.c.serial,
                    tab.c.tag_serial == tag.store_id,
                ),
            )

        try:
            row_data = conn.execute(query).one_or_none()
        except MultipleResultsFound as e:
            print(
                f"!! BackgroundModel.build(): multiple results found when querying for BackgroundModel"
            )
            raise e

        if row_data is None:
            # build and return an unpopulated object
            return BackgroundModel(
                payload=None,
                solver_labels=solver_labels,
                cosmology=cosmology,
                atol=atol,
                rtol=rtol,
                z_sample=z_sample,
                label=label,
                tags=tags,
            )

        store_id = row_data.serial
        store_label = row_data.label

        num_expected_samples = row_data.z_samples

        # read out sample values associated with this integration
        value_table = tables["BackgroundModelValue"]

        sample_rows = conn.execute(
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
            },
            solver_labels=solver_labels,
            cosmology=cosmology,
            atol=atol,
            rtol=rtol,
            z_sample=imported_z_sample,
            label=store_label,
            tags=tags,
        )
        obj._deserialized = True
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
            "atol_serial": obj._atol.store_id,
            "rtol_serial": obj._rtol.store_id,
            "solver_serial": obj.solver.store_id,
            "z_init_serial": obj.z_sample.min.store_id,
            "z_samples": len(obj.values),
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
                    "T_photon_GeV": value.T_photon / GeV,
                    "T_photon_Kelvin": value.T_photon / Kelvin,
                    "d_lnH_dz": value.d_lnH_dz,
                    "d2_lnH_dz2": value.d2_lnH_dz2,
                    "d3_lnH_dz3": value.d3_lnH_dz3,
                    "d_wPerturbations_dz": value.d_wPerturbations_dz,
                    "d2_wPerturbations_dz2": value.d2_wPerturbations_dz2,
                },
            )

            # set store_id on behalf of the GkNumericalValue instance
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

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
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
                    atol_table.c.log10_tol.label("log10_atol"),
                    rtol_table.c.log10_tol.label("log10_rtol"),
                )
                .select_from(
                    table.join(
                        solver_table, solver_table.c.serial == table.c.solver_serial
                    )
                    .join(atol_table, atol_table.c.serial == table.c.atol_serial)
                    .join(rtol_table, rtol_table.c.serial == table.c.rtol_serial)
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
                f'       -- "{model.label}" (store_id={model.serial}, log10_atol={model.log10_atol}, log10_rtol={model.log10_rtol})'
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
                    "wkb_serial": model_serial,
                    "z_serial": z.store_id,
                    "Hubble_GeV": Hubble / GeV,
                    "rho_GeV": rho / GeV4,
                    "wBackground": wBackground,
                    "wPerturbations": wPerturbations,
                    "tau_Mpc": tau / Mpc,
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

            Hubble = row_data.Hubble_GeV * GeV
            rho = row_data.rho_GeV * GeV4
            tau = row_data.tau_Mpc * Mpc
            T_photon = row_data.T_photon_GeV * GeV

            d_lnH_dz = row_data.d_lnH_dz
            d2_lnH_dz2 = row_data.d2_lnH_dz2
            d3_lnH_dz3 = row_data.d3_lnH_dz3

            d_wPerturbations_dz = row_data.d_wPerturbations_dz
            d2_wPerturbations_dz2 = row_data.d2_wPerturbations_dz2

            if fabs(row_data.Hubble - Hubble) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored background model Hubble value (model store_id={model_serial}, z={z.store_id}) = {row_data.Hubble} differs from expected value = {Hubble}"
                )
            if fabs(row_data.wBackground - wBackground) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored background model w_Background value (model store_id={model_serial}, z={z.store_id}) = {row_data.wBackground} differs from expected value = {wBackground}"
                )

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
        )
        obj._deserialized = True
        return obj
