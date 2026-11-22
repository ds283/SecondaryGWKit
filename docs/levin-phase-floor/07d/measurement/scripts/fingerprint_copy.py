"""levin-phase-floor prompt 06 section 6, route (B): the scratch copy's store fingerprint, under the
tree on PYTHONPATH (s: its factories declare theta_residual, so it can read the copy).
``python -m runregistrykit store fingerprint`` loads the fingerprinter named by the main checkout's
runregistrykit.toml from the main checkout (28ebaf0), whose factories cannot read the copy
(out/cmd/13); this calls the same fingerprinter object, the tree's ``config.fingerprint:
fingerprinter`` (a ``runregistrykit.fingerprint.DatastoreKitFingerprinter``), with its ``take``,
and prints its ``describe`` and the digest. Read-only: datastorekit's read-only reader.

Usage: PYTHONPATH=<tree> python -B fingerprint_copy.py PRIMARY"""
import datetime
import json
import sys

import config.fingerprint as cf

primary = sys.argv[1]
print(f"# config.fingerprint from {cf.__file__}; format {cf.FINGERPRINT_FORMAT}")
fp, _inv = cf.fingerprinter.take(
    primary, {"at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
)
for line in cf.fingerprinter.describe(fp):
    print(line)
keys = sorted(k for k in fp if k not in ("classes",))
print("# top-level keys:", keys)
for k in keys:
    if k != "problems":
        print(f"{k}: {json.dumps(fp[k], default=str)[:200]}")
