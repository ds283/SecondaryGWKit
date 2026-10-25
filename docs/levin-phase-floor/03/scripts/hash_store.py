"""Hash the A3 v2 store files: SHA-256, size and mtime of the primary, the four
shards and the sidecar.  Read-only: each file is opened 'rb'.

Usage: python hash_store.py LABEL   (writes to stdout)
"""

import datetime
import hashlib
import os
import socket
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FILES = [
    "var/datastores/handover-A3-v2-lambdacdm.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm-shard0000.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm-shard0001.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm-shard0002.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm-shard0003.sqlite",
    "var/datastores/handover-A3-v2-lambdacdm.manifest.json",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"# store files {label}, {now}, host {socket.gethostname()}, HEAD {head}")
    print(f"# hash_store.py sha256 {sha256(os.path.abspath(__file__))}")
    print("# sha256  size_bytes  mtime_ns  mtime_utc  path")
    for rel in FILES:
        p = os.path.join(REPO, rel)
        st = os.stat(p)
        mt = datetime.datetime.fromtimestamp(
            st.st_mtime, datetime.timezone.utc
        ).isoformat()
        print(f"{sha256(p)}  {st.st_size}  {st.st_mtime_ns}  {mt}  {rel}")


if __name__ == "__main__":
    main()
