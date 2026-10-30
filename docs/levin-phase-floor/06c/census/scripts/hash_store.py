"""Hash the A3 v2 store files: SHA-256, size and mtime of the primary, the four shards and the
sidecar, and list any journal, -wal or -shm file beside any of them. Read-only: each file is
opened 'rb'.

Prompt 05 section 6's copy of prompt 04's hash_store.py (a record, not edited), with prompt 05's
prov.py header and the journal check added.

Usage: python hash_store.py LABEL [DIR]   (writes to stdout)

prompt 06: the optional DIR (default A3 v2's, var/datastores) hashes the same five store files in
another directory: the scratch copy of route (B), state/a3v2_copy. Changed here only by that.
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prov import STORE_DIR, header, sha256  # noqa: E402

FILES = [
    "handover-A3-v2-lambdacdm.sqlite",
    "handover-A3-v2-lambdacdm-shard0000.sqlite",
    "handover-A3-v2-lambdacdm-shard0001.sqlite",
    "handover-A3-v2-lambdacdm-shard0002.sqlite",
    "handover-A3-v2-lambdacdm-shard0003.sqlite",
    "handover-A3-v2-lambdacdm.manifest.json",
]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    store_dir = sys.argv[2] if len(sys.argv) > 2 else str(STORE_DIR)  # prompt 06
    for line in header(os.path.abspath(__file__), f"store files {label}"):
        print(line)
    print(f"# sha256  size_bytes  mtime_ns  mtime_utc  path (in {store_dir})")
    for rel in FILES:
        p = os.path.join(store_dir, rel)
        if not os.path.exists(p) and rel.endswith(".manifest.json"):  # prompt 06: the copy has none
            print(f"# {rel}: absent")
            continue
        st = os.stat(p)
        mt = datetime.datetime.fromtimestamp(
            st.st_mtime, datetime.timezone.utc
        ).isoformat()
        print(f"{sha256(p)}  {st.st_size}  {st.st_mtime_ns}  {mt}  {rel}")
    side = sorted(
        f
        for f in os.listdir(store_dir)
        if f.startswith("handover-A3-v2-lambdacdm")
        and (f.endswith("-journal") or f.endswith("-wal") or f.endswith("-shm"))
    )
    print(f"# journal/-wal/-shm files beside the store: {side if side else 'none'}")


if __name__ == "__main__":
    main()
