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
