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
