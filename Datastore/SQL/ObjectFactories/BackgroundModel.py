from typing import Optional, List

import sqlalchemy as sqla
from math import fabs
from sqlalchemy import and_, or_
from sqlalchemy.exc import MultipleResultsFound

from ComputeTargets import BackgroundModel, IntegrationSolver, BackgroundModelValue
from CosmologyConcepts import redshift_array, redshift
from CosmologyModels import BaseCosmology
from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
from MetadataConcepts import store_tag, tolerance
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
                    table.c.wkb_serial == model.store_id,
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

        compute_time = row_data.compute_time
        compute_steps = row_data.compute_steps
        RHS_evaluations = row_data.RHS_evaluations
        mean_RHS_time = row_data.mean_RHS_time
        max_RHS_time = row_data.max_RHS_time
        min_RHS_time = row_data.min_RHS_time

        solver_label = row_data.solver_label
        solver_stepping = row_data.solver_stepping
        num_expected_samples = row_data.z_samples

        solver = IntegrationSolver(
            store_id=row_data.solver_serial,
            label=solver_label,
            stepping=solver_stepping,
        )

        # read out sample values associated with this integration
        value_table = tables["BackgroundModelValue"]

        sample_rows = conn.execute(
            sqla.select(
                value_table.c.serial,
                value_table.c.z_serial,
                redshift_table.c.z,
                value_table.c.Hubble,
                value_table.c.wBackground,
                value_table.c.wPerturbations,
                value_table.c.rho,
                value_table.c.tau,
                value_table.c.d_lnH_dz,
                value_table.c.d2_lnH_dz2,
                value_table.c.d3_lnH_dz3,
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
        for row in sample_rows:
            z_value = redshift(store_id=row.z_serial, z=row.z)
            z_points.append(z_value)
            values.append(
                BackgroundModelValue(
                    store_id=row.serial,
                    z=z_value,
                    Hubble=row.Hubble,
                    wBackground=row.wBackground,
                    wPerturbations=row.wPerturbations,
                    rho=row.rho,
                    tau=row.tau,
                    d_lnH_dz=row.d_lnH_dz,
                    d2_lnH_dz2=row.d2_lnH_dz2,
                    d3_lnH_dz3=row.d3_lnH_dz3,
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
                "compute_time": compute_time,
                "compute_steps": compute_steps,
                "RHS_evaluations": RHS_evaluations,
                "mean_RHS_time": mean_RHS_time,
                "max_RHS_time": max_RHS_time,
                "min_RHS_time": min_RHS_time,
                "solver": solver,
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
            "compute_time": obj.compute_time,
            "compute_steps": obj.compute_steps,
            "RHS_evaluations": obj.RHS_evaluations,
            "mean_RHS_time": obj.mean_RHS_time,
            "max_RHS_time": obj.max_RHS_time,
            "min_RHS_time": obj.min_RHS_time,
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
        value_inserter = inserters["BackgroundModelValue"]
        for value in obj.values:
            value: BackgroundModelValue
            value_id = value_inserter(
                conn,
                {
                    "model_serial": store_id,
                    "z_serial": value.z.store_id,
                    "Hubble": value.Hubble,
                    "rho": value.rho,
                    "wBackground": value.wBackground,
                    "wPerturbations": value.wPerturbations,
                    "tau": value.tau,
                    "d_lnH_dz": value.d_lnH_dz,
                    "d2_lnH_dz2": value.d2_lnH_dz2,
                    "d3_lnH_dz3": value.d3_lnH_dz3,
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
    def validate_on_startup(conn, table, tables):
        # query the datastore for any integrations that are not validated

        atol_table = tables["tolerance"].alias("atol")
        rtol_table = tables["tolerance"].alias("rtol")
        solver_table = tables["IntegrationSolver"]
        value_table = tables["BackgroundModelValue"]

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
                sqla.Column("Hubble", sqla.Float(64), nullable=False),
                sqla.Column("wBackground", sqla.Float(64), nullable=False),
                sqla.Column("wPerturbations", sqla.Float(64), nullable=False),
                sqla.Column("rho", sqla.Float(64), nullable=False),
                sqla.Column("tau", sqla.Float(64), nullable=False),
                sqla.Column("d_lnH_dz", sqla.Float(64), nullable=False),
                sqla.Column("d2_lnH_dz2", sqla.Float(64), nullable=True),
                sqla.Column("d3_lnH_dz3", sqla.Float(64), nullable=True),
            ],
        }

    @staticmethod
    def build(payload, conn, table, inserter, tables, inserters):
        model_serial = payload["model_serial"]
        z = payload["z"]

        Hubble = payload["Hubble"]
        rho = payload["rho"]
        wBackground = payload["wBackground"]
        wPerturbations = payload["wPerturbations"]
        tau = payload["tau"]

        d_lnH_dz = payload["d_lnH_dz"]
        d2_lnH_dz2 = payload["d2_lnH_dz2"]
        d3_lnH_dz3 = payload["d3_lnH_dz3"]

        try:
            row_data = conn.execute(
                sqla.select(
                    table.c.serial,
                    table.c.Hubble,
                    table.c.rho,
                    table.c.wBackground,
                    table.c.wPerturbations,
                    table.c.tau,
                    table.c.d_lnH_dz,
                    table.c.d2_lnH_dz2,
                    table.c.d3_lnH_dz3,
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

        if row_data is None:
            store_id = inserter(
                conn,
                {
                    "wkb_serial": model_serial,
                    "z_serial": z.store_id,
                    "Hubble": Hubble,
                    "rho": rho,
                    "wBackground": wBackground,
                    "wPerturbations": wPerturbations,
                    "tau": tau,
                    "d_lnH_dz": d_lnH_dz,
                    "d2_lnH_dz2": d2_lnH_dz2,
                    "d3_lnH_dz3": d3_lnH_dz3,
                },
            )
        else:
            store_id = row_data.serial

            tau = row_data.tau
            d_lnH_dz = row_data.d_lnH_dz
            d2_lnH_dz2 = row_data.d2_lnH_dz2
            d3_lnH_dz3 = row_data.d3_lnH_dz3

            if fabs(row_data.Hubble - Hubble) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored background model Hubble value (model store_id={model_serial}, z={z.store_id}) = {row_data.Hubble} differs from expected value = {Hubble}"
                )
            if fabs(row_data.wBackground - wBackground) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored background model w_Background value (model store_id={model_serial}, z={z.store_id}) = {row_data.wBackground} differs from expected value = {wBackground}"
                )
            if fabs(row_data.Hubble - Hubble) > DEFAULT_FLOAT_PRECISION:
                raise ValueError(
                    f"Stored background model w_Perturbations value (model store_id={model_serial}, z={z.store_id}) = {row_data.wPerturbations} differs from expected value = {wPerturbations}"
                )

        return BackgroundModelValue(
            store_id=store_id,
            z=z,
            Hubble=Hubble,
            rho=rho,
            wBackground=wBackground,
            wPerturbations=wPerturbations,
            tau=tau,
            d_lnH_dz=d_lnH_dz,
            d2_lnH_dz2=d2_lnH_dz2,
            d3_lnH_dz3=d3_lnH_dz3,
        )
