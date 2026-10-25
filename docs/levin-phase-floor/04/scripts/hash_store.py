"""Hash the A3 v2 store files: SHA-256, size and mtime of the primary, the four
shards and the sidecar.  Read-only: each file is opened 'rb'.

Prompt 04's copy of 03's hash_store.py; the header records both trees' HEADs.

Usage: python hash_store.py LABEL   (writes to stdout)
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prov import MAIN, header, sha256  # noqa: E402

FILES = [
    "var/datastores/handover-A3-v2-lambdacdm.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm-shard0000.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm-shard0001.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm-shard0002.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm-shard0003.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm.manifest.json",
]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    for line in header(os.path.abspath(__file__), f"store files {label}"):
        print(line)
    print("# sha256  size_bytes  mtime_ns  mtime_utc  path")
    for rel in FILES:
        p = os.path.join(str(MAIN), rel)
        st = os.stat(p)
        mt = datetime.datetime.fromtimestamp(
            st.st_mtime, datetime.timezone.utc
        ).isoformat()
        print(f"{sha256(p)}  {st.st_size}  {st.st_mtime_ns}  {mt}  {rel}")


if __name__ == "__main__":
    main()
