"""
The polymorphic cosmology reference (prompts/datastore-generic, prompt 08).

A ``wavenumber_exit_time`` row and a ``BackgroundModel`` row name their cosmology by a serial with
no foreign key, ``cosmology_serial``, and the table that serial is in by ``cosmology_type``, one of
the identifiers of ``CosmologyModels/model_ids.py``. ``COSMOLOGY_TYPES`` maps each identifier to the
class whose table holds that cosmology. Both factories declare their reference through it, in their
``inventory_spec``, so that the map is written once and the datastore layer is told it rather than
knowing it.
"""

from typing import Dict

from CosmologyModels.model_ids import LAMBDACDM_IDENTIFIER, QCD_EOS_IDENTIFIER

# cosmology_type -> the class whose table holds that cosmology
COSMOLOGY_TYPES: Dict[int, str] = {
    LAMBDACDM_IDENTIFIER: "LambdaCDM",
    QCD_EOS_IDENTIFIER: "QCD_Cosmology",
}
