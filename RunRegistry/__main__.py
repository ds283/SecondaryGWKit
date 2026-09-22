"""`python -m RunRegistry list` — one line per run, newest first.

Run it at the start of a session and before launching anything long. A registry nobody reads is
worse than none, because it looks like coverage.
"""

import argparse
import sys
import time

from . import DEFAULT_ROOT, DEFAULT_STALE_AFTER, list_runs


def _age(seconds) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:02.0f}m"
    return f"{seconds // 86400:.0f}d {(seconds % 86400) // 3600:02.0f}h"


def _progress(entry) -> str:
    done, total = entry["units_done"], entry["units_total"]
    if total is None and not done:
        # Nothing to count. A job that declares no total and has recorded no unit is either
        # reporting a stage instead (a pipeline run, which has no denominator) or has not got to
        # its first unit; `0/?` would read as "no work done", which is more than it knows.
        return "-"
    return f"{'?' if done is None else done}/{'?' if total is None else total}"


def _purpose(entry) -> str:
    if entry["purpose"]:
        text = entry["purpose"]
    elif entry["has_manifest"]:
        text = "(manifest carries no purpose)"
    else:
        text = "(no manifest — predates the registry, or was not begun through it)"
    stage = entry.get("stage")
    if stage:
        # a job with no total to count against says where it has got to instead; see
        # `Run.heartbeat`. It goes here rather than in PROGRESS because it is a sentence,
        # and because a stage in a column of fractions would read as one.
        text += f"  · {stage[:72]}"
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="RunRegistry", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lister = sub.add_parser("list", help="every run, newest first")
    lister.add_argument("--root", default=DEFAULT_ROOT, help=f"default {DEFAULT_ROOT}")
    lister.add_argument(
        "--stale-after",
        type=float,
        default=DEFAULT_STALE_AFTER,
        metavar="SECONDS",
        help=f"heartbeat staleness window, default {DEFAULT_STALE_AFTER:.0f}",
    )
    args = parser.parse_args(argv)

    entries = list_runs(root=args.root, stale_after=args.stale_after)
    if not entries:
        print(f"no runs under {args.root}")
        return 0

    now = time.time()
    width = max(len(entry["id"]) for entry in entries)
    print(
        f"{'':2} {'RUN':{width}}  {'STATE':9} {'PROGRESS':>9} {'AGE':>8}  PURPOSE · STAGE"
    )
    for entry in entries:
        stale = entry["liveness"] == "stale"
        print(
            f"{'!!' if stale else '':2} {entry['id']:{width}}  "
            f"{(entry['state'] or 'unknown'):9} {_progress(entry):>9} "
            f"{_age(now - entry['created_epoch']):>8}  {_purpose(entry)}"
        )

    counts = {}
    for entry in entries:
        counts[entry["liveness"]] = counts.get(entry["liveness"], 0) + 1
    summary = ", ".join(f"{n} {name}" for name, n in sorted(counts.items()))
    print(f"\n{len(entries)} run(s) under {args.root}: {summary}.")
    sys.stdout.flush()
    if counts.get("stale"):
        print(
            f"!! {counts['stale']} run(s) say they are running and are not: the pid does not "
            f"answer kill -0, or the heartbeat is older than {args.stale_after:.0f} s. Nothing "
            f"here has killed or cleaned anything — go and look.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
