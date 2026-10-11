"""
The datastore's client registries (prompts/datastore-generic, prompt 06; README §6.2, U2).

The generic datastore layer (``datastorekit/SQL/``, ``datastorekit/store_reader.py`` and
``datastorekit/store_inventory.py``) does not import this project's registries: it is given them,
and this module is where this project keeps them.

- ``factories``: the registry of storable classes, each name mapped to its factory, in the order the
  schema is built. ``main.py`` and the extract scripts give it to ``ShardedPool``, which gives it to
  every actor; ``open_read_only`` and ``read_inventory`` are given it by their callers.
- ``serial_batch_sizes``: how many serials an actor leases from the broker at a time, per table.
  ``main.py`` gives it to ``ShardedPool``. A table not named here leases ``SerialPoolManager``'s
  default.
- ``drop_groups``: the ``--drop`` actions, each mapped to the tables it drops, and
  ``tables_to_drop``, which resolves a list of actions to tables. The groups are a client fact
  because they are the command line's vocabulary: ``main.py`` offers their names as the choices of
  ``--drop``, in this order, and gives ``ShardedPool`` only the tables. The layer drops whatever
  tables it is given in the order ``datastorekit.SQL.schema.drop_order`` derives from their foreign
  keys, so **the order of the groups, and of the tables within a group, decides nothing about the
  order of a drop**. No drop order is kept by hand.

Importing this module imports every factory, and through them the compute targets; it imports
``ray`` only as they do.
"""

from typing import Iterable, List

from Datastore.SQL.ObjectFactories.BackgroundModel import (
    sqla_BackgroundModelFactory,
    sqla_BackgroundModelTagAssociation_factory,
    sqla_BackgroundModelValue_factory,
)
from Datastore.SQL.ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericIntegration_factory,
    sqla_GkNumericValue_factory,
    sqla_GkNumericTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.GkSource import (
    sqla_GkSourceTagAssociation_factory,
    sqla_GkSource_factory,
    sqla_GkSourceValue_factory,
    sqla_GkSourceParents_factory,
)
from Datastore.SQL.ObjectFactories.GkSourcePolicy import sqla_GkSourcePolicy_factory
from Datastore.SQL.ObjectFactories.GkSourcePolicyData import (
    sqla_GkSourcePolicyData_factory,
)
from Datastore.SQL.ObjectFactories.GkWKBIntegration import (
    sqla_GkWKBIntegration_factory,
    sqla_GkWKBTagAssociation_factory,
    sqla_GkWKBValue_factory,
)
from Datastore.SQL.ObjectFactories.LambdaCDM import sqla_LambdaCDM_factory
from Datastore.SQL.ObjectFactories.OneLoopIntegral import (
    sqla_OneLoopIntegral_factory,
    sqla_OneLoopIntegralTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.QCD_Cosmology import sqla_QCDCosmology_factory
from Datastore.SQL.ObjectFactories.QuadSource import (
    sqla_QuadSource_factory,
    sqla_QuadSourceTagAssocation_factory,
    sqla_QuadSourceValue_factory,
)
from Datastore.SQL.ObjectFactories.QuadSourceIntegral import (
    sqla_QuadSourceIntegralTagAssociation_factory,
    sqla_QuadSourceIntegral_factory,
)
from Datastore.SQL.ObjectFactories.QuadSourcePolicy import sqla_QuadSourcePolicy_factory
from Datastore.SQL.ObjectFactories.TkNumericIntegration import (
    sqla_TkNumericIntegration_factory,
    sqla_TkNumericValue_factory,
    sqla_TkNumericTagAssociation_factory,
)
from Datastore.SQL.ObjectFactories.TkWKBIntegration import (
    sqla_TkWKBIntegration_factory,
    sqla_TkWKBTagAssociation_factory,
    sqla_TkWKBValue_factory,
)
from Datastore.SQL.ObjectFactories.integration_metadata import (
    sqla_IntegrationSolver_factory,
)
from Datastore.SQL.ObjectFactories.redshift import sqla_redshift_factory
from Datastore.SQL.ObjectFactories.store_tag import sqla_store_tag_factory
from Datastore.SQL.ObjectFactories.tolerance import sqla_tolerance_factory
from Datastore.SQL.ObjectFactories.version import sqla_version_factory
from Datastore.SQL.ObjectFactories.wavenumber import (
    sqla_wavenumber_factory,
    sqla_wavenumber_exit_time_factory,
)

factories = {
    "version": sqla_version_factory,
    "store_tag": sqla_store_tag_factory,
    "redshift": sqla_redshift_factory,
    "wavenumber": sqla_wavenumber_factory,
    "wavenumber_exit_time": sqla_wavenumber_exit_time_factory,
    "tolerance": sqla_tolerance_factory,
    "LambdaCDM": sqla_LambdaCDM_factory,
    "QCD_Cosmology": sqla_QCDCosmology_factory,
    "IntegrationSolver": sqla_IntegrationSolver_factory,
    "BackgroundModel": sqla_BackgroundModelFactory,
    "BackgroundModel_tags": sqla_BackgroundModelTagAssociation_factory,
    "BackgroundModelValue": sqla_BackgroundModelValue_factory,
    "TkNumericIntegration": sqla_TkNumericIntegration_factory,
    "TkNumeric_tags": sqla_TkNumericTagAssociation_factory,
    "TkNumericValue": sqla_TkNumericValue_factory,
    "TkWKBIntegration": sqla_TkWKBIntegration_factory,
    "TkWKB_tags": sqla_TkWKBTagAssociation_factory,
    "TkWKBValue": sqla_TkWKBValue_factory,
    "GkNumericIntegration": sqla_GkNumericIntegration_factory,
    "GkNumeric_tags": sqla_GkNumericTagAssociation_factory,
    "GkNumericValue": sqla_GkNumericValue_factory,
    "GkWKBIntegration": sqla_GkWKBIntegration_factory,
    "GkWKB_tags": sqla_GkWKBTagAssociation_factory,
    "GkWKBValue": sqla_GkWKBValue_factory,
    "GkSourcePolicy": sqla_GkSourcePolicy_factory,
    "GkSourcePolicyData": sqla_GkSourcePolicyData_factory,
    "GkSource": sqla_GkSource_factory,
    "GkSource_tags": sqla_GkSourceTagAssociation_factory,
    "GkSourceValue": sqla_GkSourceValue_factory,
    # the Gk rows each GkSource is assembled from (prompts/datastore-integrity, prompt 09b)
    "GkSource_parents": sqla_GkSourceParents_factory,
    "QuadSourcePolicy": sqla_QuadSourcePolicy_factory,
    "QuadSource": sqla_QuadSource_factory,
    "QuadSource_tags": sqla_QuadSourceTagAssocation_factory,
    "QuadSourceValue": sqla_QuadSourceValue_factory,
    "QuadSourceIntegral": sqla_QuadSourceIntegral_factory,
    "QuadSourceIntegral_tags": sqla_QuadSourceIntegralTagAssociation_factory,
    "OneLoopIntegral": sqla_OneLoopIntegral_factory,
    "OneLoopIntegral_tags": sqla_OneLoopIntegralTagAssociation_factory,
}

drop_groups = {
    "tk-numeric": ["TkNumericIntegration", "TkNumeric_tags", "TkNumericValue"],
    "tk-wkb": ["TkWKBIntegration", "TkWKB_tags", "TkWKBValue"],
    "quad-source": ["QuadSource", "QuadSource_tags", "QuadSourceValue"],
    "gk-numeric": ["GkNumericIntegration", "GkNumeric_tags", "GkNumericValue"],
    "gk-wkb": ["GkWKBIntegration", "GkWKB_tags", "GkWKBValue"],
    "gk-source": ["GkSource", "GkSource_tags", "GkSourceValue", "GkSource_parents"],
    "gk-source-policy-records": ["GkSourcePolicy"],
    "gk-source-policy": ["GkSourcePolicyData"],
    "quad-source-integral": [
        "QuadSourceIntegral",
        "QuadSourceIntegral_tags",
    ],
    "1loop-integral": [
        "OneLoopIntegral",
        "OneLoopIntegral_tags",
    ],
}

serial_batch_sizes = {
    "store_tag": 5,
    "redshift": 200,
    "wavenumber": 100,
    "wavenumber_exit_time": 100,
    "tolerance": 5,
    "LambdaCDM": 5,
    "QCD_Cosmology": 5,
    "IntegrationSolver": 6,
    "BackgroundModel": 5,
    "BackgroundModelValue": 100,
    "TkNumericIntegration": 200,
    "TkNumeric_tags": 1000,
    "TkNumericValue": 7500,
    "TkWKBIntegration": 200,
    "TkWKB_tags": 1000,
    "TkWKBValue": 7500,
    "GkNumericIntegration": 200,
    "GkNumeric_tags": 1000,
    "GkNumericValue": 7500,
    "GkWKBIntegration": 200,
    "GkWKB_tags": 1000,
    "GkWKBValue": 7500,
    "GkSourcePolicy": 5,
    "GkSourcePolicyData": 50,
    "GkSource": 200,
    "GkSource_tags": 1000,
    "GkSourceValue": 7500,
    # one row per z_source of each GkSource, as GkSourceValue has (prompts/datastore-integrity,
    # prompt 09b)
    "GkSource_parents": 7500,
    "QuadSource": 200,
    "QuadSourcePolicy": 5,
    "QuadSource_tags": 1000,
    "QuadSourceValue": 7500,
    "QuadSourceIntegral": 10000,
    "QuadSourceIntegral_tags": 3000,
    "OneLoopIntegral": 10000,
    "OneLoopIntegral_tags": 3000,
}


def tables_to_drop(actions: Iterable[str]) -> List[str]:
    """
    The tables of the drop groups named by ``actions``, each once, in the order the actions and
    then each group's list give them. An action that is not a drop group raises ``ValueError``
    naming it, and the groups there are.
    """
    tables: List[str] = []
    for action in actions:
        if action not in drop_groups:
            raise ValueError(
                f'Unknown drop action "{action}": the drop actions are {list(drop_groups)}'
            )
        for table in drop_groups[action]:
            if table not in tables:
                tables.append(table)
    return tables
