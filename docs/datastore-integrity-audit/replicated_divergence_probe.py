"""
Probe: which divergence modes of a replicated class does the structured inventory
(Datastore/store_inventory.py read_inventory) name, and which does it not?

Builds throwaway full stores with Datastore/tests/real_store_fixtures.build_full_store in a
temporary directory under this directory, applies one divergence per case with raw SQL on one
shard, and prints the named problems. No Ray, no real datastore.

Run from the repository root:
    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_divergence_probe.py
"""

import tempfile
from pathlib import Path

from Datastore.store_inventory import read_inventory
from Datastore.tests.real_store_fixtures import build_full_store

HERE = Path(__file__).resolve().parent

CASES = {
    "baseline": {},
    # the same tolerance row under a different serial on shard 2 (replicated-only shard), with the
    # replicated references on that shard following it: what "K->s on one shard, K->s' on another"
    # looks like when the shard is internally consistent
    "serial-only, shard 2, refs followed": {
        2: [
            "UPDATE tolerance SET serial = 903 WHERE serial = 3",
            "UPDATE wavenumber_exit_time SET atol_serial = 903 WHERE atol_serial = 3",
            "UPDATE wavenumber_exit_time SET rtol_serial = 903 WHERE rtol_serial = 3",
        ]
    },
    # tolerance 1, which sharded rows on shard 1 reference: relabelled on shard 1 with its
    # replicated references following, the sharded rows keeping the controller's serial
    "serial-only tolerance 1, shard 1, sharded refs left": {
        1: [
            "UPDATE tolerance SET serial = 901 WHERE serial = 1",
            "UPDATE wavenumber_exit_time SET atol_serial = 901 WHERE atol_serial = 1",
        ]
    },
    # the same, with the sharded references on shard 1 following too
    "serial-only tolerance 1, shard 1, all refs followed": {
        1: [
            "UPDATE tolerance SET serial = 901 WHERE serial = 1",
            "UPDATE wavenumber_exit_time SET atol_serial = 901 WHERE atol_serial = 1",
            "UPDATE TkNumericIntegration SET atol_serial = 901 WHERE atol_serial = 1",
            "UPDATE GkNumericIntegration SET atol_serial = 901 WHERE atol_serial = 1",
            "UPDATE QuadSourceIntegral SET atol_serial = 901 WHERE atol_serial = 1",
        ]
    },
    # a duplicate: a second copy of tolerance 3 under a new serial on shard 2 only (what a store
    # replayed onto a shard that already holds the key would leave, if it did not look up first)
    "duplicate row on shard 2": {
        2: [
            "INSERT INTO tolerance (serial, timestamp, log10_tol) "
            "SELECT 933, timestamp, log10_tol FROM tolerance WHERE serial = 3"
        ]
    },
    # an unreferenced replicated row under a different serial (store_tag 5 relabelled everywhere
    # on shard 2, including its association rows)
    "serial-only store_tag, shard 2": {
        2: [
            "UPDATE store_tag SET serial = 905 WHERE serial = 5",
            "UPDATE BackgroundModel_tags SET tag_serial = 905 WHERE tag_serial = 5",
        ]
    },
    # what the real write path does on every replica: BackgroundModelValue rows get the replica's
    # own leased serials (BackgroundModel.py:664 passes no serial)
    "BackgroundModelValue serials differ, shard 1": {
        1: ["UPDATE BackgroundModelValue SET serial = serial + 1000"]
    },
    "missing replicated row, shard 2": {2: ["DELETE FROM LambdaCDM WHERE serial = 2"]},
    "validated flag differs, shard 2": {
        2: ["UPDATE BackgroundModel SET validated = 0 WHERE serial = 1"]
    },
    "partial value rows, shard 2": {
        2: ["DELETE FROM BackgroundModelValue WHERE serial = 5"]
    },
    "timestamp only differs, shard 2": {
        2: ["UPDATE redshift SET timestamp = '2001-01-01 00:00:00.000000'"]
    },
}


def main():
    with tempfile.TemporaryDirectory(dir=HERE, prefix="probe-") as tmp:
        root = Path(tmp)
        for n, (label, sql) in enumerate(CASES.items()):
            d = root / f"s{n}"
            d.mkdir()
            store = build_full_store(d, shards=3, extra_sql=sql)
            inv = read_inventory(store.primary)
            print(f"== {label}")
            if len(inv.problems) == 0:
                print("   (no problem named)")
            for p in inv.problems:
                print("   " + p[:220])


if __name__ == "__main__":
    main()
