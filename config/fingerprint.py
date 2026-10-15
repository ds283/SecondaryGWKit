"""
The store fingerprint's client side (runregistry-extraction, prompt 06; README §6.2, C17; the
user's decision D3).

A store's fingerprint is split by layer. **What goes into it** is this project's: the
``inventory_spec()`` on each factory, the registry of factories, and the format number below.
**How that content is encoded and digested** is DatastoreKit's (``datastorekit.store_fingerprint``),
versioned there by its own ``ENCODING_FORMAT``. **When a fingerprint is taken and where it is
recorded** is RunRegistryKit's (``runregistrykit.stores``), which is handed the fingerprinter below
and treats what it returns as opaque. This module is where this project keeps its part, so that the
format number sits beside the thing it versions.

- ``FINGERPRINT_FORMAT``: this project's fingerprint format, with its history.
- ``fingerprinter``: a ``runregistrykit.fingerprint.DatastoreKitFingerprinter`` over
  ``config.datastore.factories`` at ``FINGERPRINT_FORMAT``. ``runregistrykit.toml`` at the
  repository root names it (``[stores] fingerprinter = "config.fingerprint:fingerprinter"``), and
  every ``runregistrykit`` store operation loads it from there. It is an instance, as the loader
  requires.

Importing this module loads nothing that ``config.datastore`` does not already load:
``DatastoreKitFingerprinter`` imports DatastoreKit only inside its methods.
"""

from runregistrykit.fingerprint import DatastoreKitFingerprinter

from config.datastore import factories

# the fingerprint's format. A fingerprint is compared only with one of the same format; a change to
# what the inventory records, or to how a digest is formed, bumps this and captures a new golden
# fingerprint in the same commit, beside the old ones and never over them
# (`RunRegistry/tests/data/full_store_fingerprint*.json`; `test_store_fingerprint.GOLDEN` names the
# current one). Format 2: prompts/datastore-integrity prompt 07 put the WKB rows' numeric parent in
# their key. Format 3: prompt 08 put the QuadSourcePolicy in the QuadSourceIntegral rows' key.
# Format 4: prompt 09a put the two TkNumericIntegrations, Tq and Tr, in the QuadSource rows' key.
# Format 5: prompt 09b put the set of Gk numeric and WKB rows each GkSource is assembled from in its
# key, resolved through GkSource_parents to those rows' own identities. Format 6: prompt 09c put the
# six stored rows each QuadSourceIntegral is computed from -- its QuadSource, its
# GkSourcePolicyData, and the Tk numeric and WKB rows for q and r -- in its key, each resolved to its
# own identity, across shards where it is held on another
#
# Since runregistry-extraction prompt 06 (2026-10-15), the history above stands here verbatim,
# moved from the registry package's `stores.py`, which that prompt deleted. The golden files are
# now under `ObjectFactories/tests/data/`, and `test_store_fingerprint_golden.GOLDEN` names the
# current one. The encoding and the digest are DatastoreKit's, versioned by its `ENCODING_FORMAT`
# and recorded beside the digest as `encoding_format`; a change there does not bump this number. A
# change to what the inventory records still does, with a new golden file beside the old ones.
FINGERPRINT_FORMAT = 6

fingerprinter = DatastoreKitFingerprinter(factories, FINGERPRINT_FORMAT)
