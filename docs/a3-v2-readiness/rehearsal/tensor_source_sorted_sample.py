"""NOT THE SHIPPED READER. `extract_tensor_source_data.py`, unmodified, with its `sample` sorted.

A verification wrapper for a3-v2-readiness prompt 07 (§2.3), written under the user's decision of
2026-10-04. The shipped reader draws its k subsample with `random.sample`, which returns the k in
random order, and then pairs them with `itertools.combinations_with_replacement`. A pair with q > r
matches no stored `QuadSource` (the store holds q <= r only), its labels-only lookup returns an
unpopulated object, and the remote plot fails on `values has not yet been populated`
(`[07-the-tensor-source-reader-pairs-k-in-random-order-and-crashes-on-q-gt-r]`). With three k the
shipped reader passes only when the draw happens to be ascending, 1 time in 6.

This wrapper replaces `random.sample` before the script imports it (`from random import sample`),
with a function that calls the real `sample` and, **only** when every item drawn is a
`wavenumber_exit_time`, returns the items in ascending k. Any other call gets the real result,
unchanged. Then it runs the unmodified script with `runpy.run_path(..., run_name="__main__")` and
the given arguments. Nothing else is changed.

It shows only that the rest of the reader's read path works, read-only, on every `QuadSource` row
of the store. It does not show that the shipped reader works: it does not.

    PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/tensor_source_sorted_sample.py \\
        --database <store> --ray-address local --models LambdaCDM --output <dir>
"""

import random
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "extract_tensor_source_data.py"

_real_sample = random.sample


def _k_of(item):
    return item.k.k_inv_Mpc


def sorted_sample(population, k, **kwargs):
    drawn = _real_sample(population, k, **kwargs)
    from CosmologyConcepts.wavenumber import wavenumber_exit_time

    if len(drawn) > 0 and all(isinstance(x, wavenumber_exit_time) for x in drawn):
        before = [_k_of(x) for x in drawn]
        drawn = sorted(drawn, key=_k_of)
        print(
            "** tensor_source_sorted_sample (NOT THE SHIPPED READER): random.sample drew k = "
            f"{before} /Mpc; returned in ascending order {[_k_of(x) for x in drawn]} /Mpc"
        )
    return drawn


def main(argv) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    random.sample = sorted_sample
    sys.argv = [str(SCRIPT)] + list(argv)
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
