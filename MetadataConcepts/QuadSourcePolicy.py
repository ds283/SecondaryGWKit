from typing import Optional

from Datastore import DatastoreObject
from config.defaults import DEFAULT_LEVIN_THRESHOLD, DEFAULT_QUADSOURCE_NUMERIC_POLICY

_allowed_numeric_policies = ["maximize_numeric", "maximize-WKB"]


class QuadSourcePolicy(DatastoreObject):
    """
    Persisted configuration object for the quadratic source time integral, the QuadSource
    analogue of GkSourcePolicy.

    **Nothing reads it.** It is created in main.py and threaded through run_pipeline, but
    neither ComputeTargets/QuadSource.py nor ComputeTargets/QuadSourceIntegral.py consults
    either field, and this is deliberate as of prompt 10 of prompts/source-remediation:

      * `numeric_policy` would choose a numeric/WKB crossover inside an overlap region, as
        GkSourcePolicy.numeric_policy does for the Green's function. The transfer function has
        no overlap: main.py integrates TkNumericIntegration in mode="stop" down to a phase
        minimum and starts TkWKBIntegration at exactly that redshift, so the hand-over is a
        single, already-determined redshift recoverable from the stored objects
        (ComputeTargets/TkSourceFunctions.py). There is no choice left to make.

      * `Levin_threshold` would be a minimum oscillation rate below which a sub-interval of the
        source integral is sent to ordinary quadrature instead of Levin quadrature. Since prompt
        08 (audit 2026-09 A4/QI-6) every sub-interval carrying an oscillatory factor is handed to
        adaptive_levin_sincos, whose own total-variation gate decides per region whether to use
        the Levin rule or Clenshaw-Curtis. A threshold here would duplicate that decision using
        less information (it cannot see the composed phase theta_G +/- theta_q +/- theta_r).
        Prompt 08 section 6 measured the cost of relying on the driver's gate in the weakly
        oscillatory regime -- 2.65-3.56x the wall-clock of the old direct quadrature, for
        1.00-1.44x the integrand evaluations, the excess being per-region driver overhead -- and
        the author's decision was to accept it and improve the driver's choice of fallback in
        AdaptiveLevin/ rather than to reinstate a threshold here.

    The object stays persisted and threaded so that the schema and the run signatures do not
    churn, and so that a future policy for this stage has somewhere to live. Do not read
    `Levin_threshold` from a compute target without revisiting the reasoning above.
    """

    def __init__(
        self,
        store_id: int,
        Levin_threshold: int = DEFAULT_LEVIN_THRESHOLD,
        numeric_policy: str = DEFAULT_QUADSOURCE_NUMERIC_POLICY,
        label: Optional[str] = None,
    ):
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        self._label = label

        self._Levin_threshold = Levin_threshold

        if numeric_policy not in _allowed_numeric_policies:
            numeric_policy = DEFAULT_QUADSOURCE_NUMERIC_POLICY
            print(
                f'!! Unknown Quadsource numeric policy "{numeric_policy}". Defaulting to policy "{DEFAULT_QUADSOURCE_NUMERIC_POLICY}"'
            )
        self._numeric_policy = numeric_policy

    @property
    def Levin_threshold(self) -> float:
        return self._Levin_threshold

    @property
    def numeric_policy(self) -> str:
        return self._numeric_policy

    @property
    def label(self) -> Optional[str]:
        return self._label
