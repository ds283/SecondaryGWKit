from math import log10, pow

from Datastore import DatastoreObject


class tolerance(DatastoreObject):
    def __init__(self, store_id: int, **kwargs):
        """
        Construct a datastore-backed object representing a tolerance (absolute or relative).
        Effectively tokenizes a floating point number to an integer.
        :param store_id: unique Datastore id. Should not be None
        :param tol: tolerance
        """
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        if "log10_tol" in kwargs:
            log10_tol = kwargs["log10_tol"]
            self.log10_tol = log10_tol
            self.tol = pow(10.0, log10_tol)
        elif "tol" in kwargs:
            tol = kwargs["tol"]
            self.tol = tol
            self.log10_tol = log10(tol)
        else:
            raise RuntimeError(
                'Neither "tol" nor "log10_tol" was supplied to tolerance() constructor'
            )

    def __float__(self):
        """
        Cast to float.
        :return:
        """
        return self.tol
