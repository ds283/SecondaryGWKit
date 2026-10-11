"""Compare closed stores' fingerprints with fingerprints recorded by earlier runs.

    PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \\
        var/datastores/<store>.sqlite [...] --against var/runs/<run id> [...]

Each store is fingerprinted read-only by `RunRegistry.stores.fingerprint_store(write=False)`, which
reads it through `read_inventory`, `mode=ro` and with no Ray, and writes nothing. Each `--against`
argument is a run directory whose `status.json` carries a `fingerprint` (its run's fingerprint at
finish), or a path to such a `status.json`. Every store is compared with every recorded fingerprint
by `compare_fingerprints(recorded, current)`, which never compares `taken`. A store's comparison
with its own sidecar is printed as well.

It is how a rehearsal's stores are checked against an earlier rehearsal whose stores are retired
but whose run directories remain: R0 against a3-v2-readiness 07's reference, and R1 and R2 against
R0 (dk-and-rpk-migration README §4.1).

Exit status: 0 if every comparison is empty, every store is at the current format with no problem,
and each matches its sidecar; 1 otherwise; 2 on a usage error.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RunRegistry.stores import (  # noqa: E402
    FINGERPRINT_FORMAT,
    compare_fingerprints,
    fingerprint_store,
)


def recorded(path: Path):
    status = path / "status.json" if path.is_dir() else path
    data = json.loads(status.read_text())
    if "fingerprint" not in data:
        raise SystemExit(f"{status} records no fingerprint")
    return status, data["fingerprint"]


def main(argv) -> int:
    if "--against" not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    at = argv.index("--against")
    stores, against = argv[:at], argv[at + 1 :]
    if not stores or not against:
        print(__doc__, file=sys.stderr)
        return 2
    records = [recorded(Path(p)) for p in against]
    ok = True
    for status, fp in records:
        taken = fp.get("taken", {})
        print(
            f"recorded: {status}\n  format {fp.get('fingerprint_format')}, digest {fp.get('digest')}, "
            f"{len(fp.get('classes', {}))} classes, problems {fp.get('problems') or 'none'}; "
            f"taken by {taken.get('run_id')} at {taken.get('git_head')} "
            f"(git_dirty {taken.get('git_dirty')})"
        )
    for primary in stores:
        result = fingerprint_store(primary, write=False)
        fp = result["fingerprint"]
        print(f"\nstore: {result['primary']}")
        print(
            f"  format {fp['fingerprint_format']}, digest {fp['digest']}, "
            f"{len(fp['classes'])} classes, problems {fp['problems'] or 'none'}"
        )
        if fp["fingerprint_format"] != FINGERPRINT_FORMAT or fp["problems"]:
            ok = False
        if result["comparison"] is None:
            print("  against its sidecar: no fingerprint recorded")
            ok = False
        else:
            print(f"  against its sidecar: {len(result['comparison'])} difference(s)")
            ok = ok and not result["comparison"]
        for status, rec in records:
            diffs = compare_fingerprints(rec, fp)
            print(
                f"  compare_fingerprints({status}, store): {len(diffs)} difference(s)"
            )
            for d in diffs:
                print(f"    {d['text']}")
            ok = ok and not diffs
    print(f"\nverdict: {'equal' if ok else 'DIFFERENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
