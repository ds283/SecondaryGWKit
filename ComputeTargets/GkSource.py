from typing import Optional, List

from ComputeTargets import BackgroundModel
from CosmologyConcepts import wavenumber_exit_time, redshift, wavenumber, redshift_array
from Datastore import DatastoreObject
from MetadataConcepts import store_tag
from utilities import check_units


class GkSource(DatastoreObject):
    def __init__(
        self,
        payload,
        model: BackgroundModel,
        k: wavenumber_exit_time,
        z_response: redshift,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, model.cosmology)

        self._z_sample = z_sample

        self._model = model
        self._k_exit = k
        self._z_response = z_response

        self._label = label
        self._tags = tags if tags is not None else []

        self._compute_ref = None

        if "Gk_data" in payload:
            DatastoreObject.__init__(self, None)

        elif "store_id" in payload:
            DatastoreObject.__init__(self, payload["store_id"])

        else:
            raise RuntimeError(
                'GkSource: did not find either "Gk_data" or "store_id" in payload'
            )

        if self._z_sample is not None:
            z_response_float = float(z_response)
            # check that each source redshift is earlier than the specified response redshift
            for z in self._z_sample:
                z_float = float(z)

                if z_float < z_response_float:
                    raise ValueError(
                        f"GkSource: source redshift sample point z={z_float:.5g} exceeds response redshift z={z_response_float:.5g}"
                    )


class GkSourceValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z_source: redshift,
        G: Optional[float] = None,
        Gprime: Optional[float] = None,
        theta: Optional[float] = None,
        H_ratio: Optional[float] = None,
        sin_coeff: Optional[float] = None,
        cos_coeff: Optional[float] = None,
        G_WKB: Optional[float] = None,
        omega_WKB_sq: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z_source = z_source

        has_numeric = any([G is not None, Gprime is not None])
        has_WKB = any(
            [
                theta is not None,
                H_ratio is not None,
                sin_coeff is not None,
                cos_coeff is not None,
                G_WKB,
            ]
        )

        all_numeric = all([G is not None, Gprime is not None])
        all_WKB = all(
            [
                theta is not None,
                H_ratio is not None,
                sin_coeff is not None,
                cos_coeff is not None,
                G_WKB,
            ]
        )

        if has_numeric and has_WKB:
            raise ValueError(
                "GkSourceValue: both numerical and WKB values specified. Only one should be supplied."
            )

        if not has_numeric and not has_WKB:
            raise ValueError(
                "GkSourceValue: neither numerical nor WKB values specified."
            )

        if has_numeric and not all_numeric:
            raise ValueError(
                "GkSourceValue: some numerical data was not supplied. G and Gprime should both be specified."
            )

        if has_WKB and not all_WKB:
            raise ValueError(
                "GkSourceValue: some WKB data was not supplied. theta, H_ratio, sin_coeff, cos_coeff and G_WKB should all be specified."
            )

        if has_numeric:
            self.type = "numeric"
        else:
            self.type = "WKB"

        self._G = G
        self._Gprime = Gprime

        self._theta = theta
        self._H_ratio = H_ratio
        self._sin_coeff = sin_coeff
        self._cos_coeff = cos_coeff
        self._G_WKB = G_WKB

        self._omega_WKB = omega_WKB_sq
