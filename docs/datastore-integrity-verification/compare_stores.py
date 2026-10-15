"""D10-2's comparison (prompt 10 §2.1): fingerprint both closed stores read-only, compare them,
and print the per-class counts beside each other.

    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/compare_stores.py \\
        var/datastores/datastore-integrity-10-reference.sqlite \\
        var/datastores/datastore-integrity-10-resumed.sqlite

`fingerprint_store(write=False)` reads each store through `read_inventory`, `mode=ro` and with no
Ray, and compares it with the fingerprint its sidecar records. `compare_fingerprints(reference,
resumed)` must be empty. It writes nothing; the exit status is 0 only if the comparison is empty,
both are at the current format, neither has a problem, and each matches its sidecar.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datastorekit.store_fingerprint import compare_fingerprints  # noqa: E402
from runregistrykit.stores import fingerprint_store  # noqa: E402

from config.fingerprint import FINGERPRINT_FORMAT  # noqa: E402


def main(reference, resumed) -> int:
    taken = {}
    ok = True
    for name, primary in (("reference", reference), ("resumed", resumed)):
        result = fingerprint_store(primary, write=False)
        fp = result["fingerprint"]
        taken[name] = fp
        print(f"{name}: {result['primary']}")
        print(f"  format {fp['fingerprint_format']}, digest {fp['digest']}")
        print(f"  problems {fp['problems'] or 'none'}")
        if result["comparison"] is None:
            print("  sidecar: no fingerprint recorded")
            ok = False
        else:
            print(
                f"  against its sidecar: {len(result['comparison'])} difference(s)"
                + ("" if result["comparison"] else ", it matches")
            )
            ok = ok and not result["comparison"]
        ok = (
            ok and fp["fingerprint_format"] == FINGERPRINT_FORMAT and not fp["problems"]
        )

    differences = compare_fingerprints(taken["reference"], taken["resumed"])
    print()
    print(f"compare_fingerprints(reference, resumed): {len(differences)} difference(s)")
    for entry in differences:
        print(f"  {entry['kind']}: {entry['class']}: {entry['text']}")
    ok = ok and not differences

    print()
    width = max(len(name) for name in taken["reference"]["classes"])
    print(f"  {'class':<{width}}  {'reference':>9}  {'resumed':>9}  digests")
    names = list(taken["reference"]["classes"])
    names += [n for n in taken["resumed"]["classes"] if n not in names]
    for name in names:
        a = taken["reference"]["classes"].get(name)
        b = taken["resumed"]["classes"].get(name)
        count_a = a["count"] if a else "-"
        count_b = b["count"] if b else "-"
        same = "equal" if a and b and a["digest"] == b["digest"] else "DIFFER"
        print(f"  {name:<{width}}  {count_a:>9}  {count_b:>9}  {same}")
    print()
    print("verdict:", "equal" if ok else "NOT equal")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
