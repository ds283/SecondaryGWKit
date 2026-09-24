#!/usr/bin/env python3
"""
Copy or move a closed ShardedPool datastore under a new name.

    python tools/sharded_store.py copy SRC DST
    python tools/sharded_store.py move SRC DST

SRC and DST are primary files (foo.sqlite). Every file is renamed to DST's stem: the shards become
<DST stem>-shard0000<suffix>, ..., and the destination primary's `shards` table is rewritten to
name them. This script calls ShardedPool.copy_store / ShardedPool.move_store and nothing else. On
success it prints the destination's shard map; on refusal or failure it prints the reason and
exits non-zero. It never deletes or overwrites a file, and after a failure it does not clean up.

Three limits, by design:

* It handles the primary and its shards and nothing else. Any other file beside the store stays
  where it is; in particular a <stem>.manifest.json is neither copied nor moved.
* It cannot tell whether a process has the store open. These stores use SQLite's rollback
  journal, which leaves no file while the store is idle. Checking that nothing is running against
  the store is the caller's job.
* A registry-level tool is the place that does both: carries the store's manifest with it, and
  checks that no running job names the store (prompts/datastore-portability/README.md
  sections 6.3-6.4). This script does not consult the registry.

It imports ShardedPool, and so ray, but never initialises Ray. It puts its own repository root on
sys.path, so it runs from any directory with no PYTHONPATH.
"""

import argparse
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Datastore.SQL.ShardedPool import ShardedPool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sharded_store.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "operation",
        choices=["copy", "move"],
        help="copy leaves the source as it was; move renames its files",
    )
    parser.add_argument("src", help="primary file of the existing, closed store")
    parser.add_argument(
        "dst",
        help="primary file of the new store; it and its shard names must not exist",
    )
    return parser


def main(argv: List[str]) -> int:
    args = _parser().parse_args(argv)
    operation = (
        ShardedPool.copy_store if args.operation == "copy" else ShardedPool.move_store
    )
    try:
        shard_map = operation(args.src, args.dst)
    except RuntimeError as e:
        print(f"!! {e}", file=sys.stderr)
        return 1

    print(
        f'>> {args.operation}: "{Path(args.dst).resolve()}" with {len(shard_map)} shards'
    )
    for serial, path in sorted(shard_map.items()):
        print(f">>   shard #{serial}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
