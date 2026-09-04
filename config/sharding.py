from CosmologyConcepts import wavenumber, wavenumber_exit_time

replicated_tables = [
    "version",
    "store_tag",
    "redshift",
    "wavenumber",
    "wavenumber_exit_time",  # needs to be replicated so that we can batch query QuadSourceIntegral and OneLoopIntegral objects
    "tolerance",
    "LambdaCDM",
    "QCD_Cosmology",
    "IntegrationSolver",
    "BackgroundModel",
    "BackgroundModelValue",
    "GkSourcePolicy",
    "QuadSourcePolicy",
]

sharded_tables = {
    "TkNumericIntegration": "k",
    "TkNumericValue": "k",
    "TkWKBIntegration": "k",
    "TkWKBValue": "k",
    "QuadSource": "q",
    "QuadSourceValue": "q",
    "GkNumericIntegration": "k",
    "GkNumericValue": "k",
    "GkWKBIntegration": "k",
    "GkWKBValue": "k",
    "GkSource": "k",
    "GkSourceValue": "k",
    "GkSourcePolicyData": "k",
    "QuadSourceIntegral": "k",
    "OneLoopIntegral": "k",
}

read_table_config = {
    "wavenumber": {"tables_arg": False},
    "redshift": {"tables_arg": True},
}


# Merge policies for pool.inventory() calls on sharded tables.
# Each field in the factory's inventory() return value needs a merge policy:
#   lists/sets → "extend"
#   datetimes  → "earliest" or "latest"
#   numbers    → "sum" (also "min"/"max")
#
# Only sharded classes go in inventory_config. Replicated classes are served
# from a single shard and are never merged, so an entry for one here would be
# harmless but misleading.

_compute_target_merge = {
    "validated": {
        "labels": "extend",
        "earliest_timestamp": "earliest",
        "latest_timestamp": "latest",
    },
    "unvalidated": {
        "labels": "extend",
        "earliest_timestamp": "earliest",
        "latest_timestamp": "latest",
    },
}

_no_validated_merge = {
    "count": "sum",
    "earliest_timestamp": "earliest",
    "latest_timestamp": "latest",
}

_value_table_merge = {"count": "sum"}

inventory_config = {
    # Group A: compute targets with a validated/unvalidated split
    "TkNumericIntegration": _compute_target_merge,
    "TkWKBIntegration": _compute_target_merge,
    "QuadSource": _compute_target_merge,
    "GkNumericIntegration": _compute_target_merge,
    "GkWKBIntegration": _compute_target_merge,
    "GkSource": _compute_target_merge,
    # Group B: compute targets with no validated column, and potentially
    # numerous, so no label list is reported
    "GkSourcePolicyData": _no_validated_merge,
    "QuadSourceIntegral": _no_validated_merge,
    "OneLoopIntegral": _no_validated_merge,
    # Group C: high-volume value tables, "timestamp": False -- count only
    "TkNumericValue": _value_table_merge,
    "TkWKBValue": _value_table_merge,
    "QuadSourceValue": _value_table_merge,
    "GkNumericValue": _value_table_merge,
    "GkWKBValue": _value_table_merge,
    "GkSourceValue": _value_table_merge,
}


shard_key_type = wavenumber


# get wavenumber store id for a wavenumber object, interpreted as a shard key,
# or a proxy for it
def shard_key_wavenumber_store_id(obj):
    if isinstance(obj, wavenumber):
        return obj.store_id

    if isinstance(obj, wavenumber_exit_time):
        return obj.k.store_id

    raise RuntimeError(
        f'Could not determine wavenumber shard key for object of type "{type(obj)}"'
    )
