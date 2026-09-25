"""`python -m RunRegistry list` — one line per run, newest first.

Run it at the start of a session and before launching anything long. A registry nobody reads is
worse than none, because it looks like coverage.

`python -m RunRegistry store {show,create,adopt,copy,move,fingerprint}` manages a datastore and
its `<stem>.manifest.json` sidecar (`RunRegistry.stores`). `show` is read-only. `create` and
`adopt` write a sidecar and never open the store. `copy` and `move` move the store's files with
`ShardedPool` and carry the sidecar, and refuse a store that any `running` run names, alive or
stale. `fingerprint` reads the closed store read-only, compares its content fingerprint with the
one the sidecar records, and writes it into the sidecar only with `--write`; it refuses a store a
`running` run names, and `--listing` writes the full listing to a new file away from the store.
None of them initialises Ray, and none deletes anything.
"""

import argparse
import json
import sys
import time

from . import DEFAULT_ROOT, DEFAULT_STALE_AFTER, list_runs
from . import stores


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


def _show(args) -> int:
    reading = stores.read_sidecar(args.primary)
    print(f"sidecar:  {reading.path}")
    print(f"kind:     {reading.kind}")
    if reading.problems:
        print("problems:")
        for problem in reading.problems:
            print(f"  !! {problem}")
    else:
        print("problems: none")
    if reading.fields is not None and "datastore" in reading.fields:
        value = reading.fields["datastore"]
        if reading.legacy_path:
            print(
                f"datastore: {value!r} is a legacy path, read by its name "
                f"{stores._final_component(value)!r}, never as a path"
            )
        else:
            print(f"datastore: {value!r}")
    if reading.fields is not None:
        print("fields:")
        for line in json.dumps(reading.fields, indent=2, sort_keys=True).splitlines():
            print(f"  {line}")

    store_id = (
        (reading.fields or {}).get("store_id") if reading.kind == "registry" else None
    )
    runs = stores.runs_naming(
        [reading.primary],
        store_id if isinstance(store_id, str) else None,
        runs_root=args.runs_root,
    )
    if runs:
        print(f"runs naming this store, under {args.runs_root}:")
        for entry in runs:
            print(
                f"  {entry['id']}  {(entry['state'] or 'unknown'):9} {entry['liveness']:9} "
                f"by {' and '.join(entry['matched_by'])}"
            )
    else:
        print(f"runs naming this store, under {args.runs_root}: none")
    return 0


def _short(digest) -> str:
    return str(digest)[:12]


def _print_fingerprint(fingerprint) -> None:
    print(f"digest:   {fingerprint['digest']}")
    print(f"format:   {fingerprint['fingerprint_format']}")
    print("classes:  (count, digest; then each tag set)")
    width = max((len(name) for name in fingerprint["classes"]), default=0)
    for name, entry in fingerprint["classes"].items():
        print(f"  {name:{width}} {entry['count']:>8}  {_short(entry['digest'])}")
        for tag_set in entry["tag_sets"]:
            print(
                f"  {'':{width}} {tag_set['count']:>8}  {_short(tag_set['digest'])}  "
                f"[{', '.join(tag_set['tags'])}]"
            )
    if fingerprint["problems"]:
        print(
            "problems: (counts only; the store's own text is in `main.py --inventory`)"
        )
        for name, kinds in fingerprint["problems"].items():
            counted = ", ".join(f"{kind} {n}" for kind, n in kinds.items())
            print(f"  !! {name}: {counted}")
    else:
        print("problems: none")


def _fingerprint(args) -> int:
    try:
        if args.listing is not None:
            stores.check_listing_path(args.listing, args.primary)
        result = stores.fingerprint_store(
            args.primary, write=args.write, runs_root=args.runs_root
        )
    except RuntimeError as e:
        print(f"!! {e}", file=sys.stderr)
        return 1

    print(f"store:    {result['primary']}")
    _print_fingerprint(result["fingerprint"])
    recorded, comparison = result["recorded"], result["comparison"]
    if recorded is None:
        print(f"recorded: none recorded (the sidecar is {result['sidecar']})")
    elif not comparison:
        taken = recorded.get("taken") or {}
        by = f"run {taken['run_id']}" if taken.get("run_id") else "a person"
        print(
            f"recorded: matches the fingerprint taken {taken.get('when')} by {by}, at "
            f"{taken.get('git_head')}{' (dirty)' if taken.get('git_dirty') else ''}"
        )
    else:
        print(
            f"recorded: {len(comparison)} difference(s) from the recorded fingerprint:"
        )
        for entry in comparison:
            print(f"  !! {entry['text']}")

    if args.write:
        if recorded is None:
            print(">> wrote the fingerprint into the sidecar; it replaced none")
        else:
            taken = recorded.get("taken") or {}
            print(
                f">> wrote the fingerprint into the sidecar; it replaced digest "
                f"{recorded.get('digest')}, taken {taken.get('when')}"
            )
    if args.listing is not None:
        path = stores.write_listing(
            result["inventory"], result["fingerprint"], args.listing, args.primary
        )
        print(f">> listing: {path}")

    if args.write:
        return 0
    return 0 if not comparison else 1


def _store(args) -> int:
    if args.store_command == "show":
        return _show(args)
    if args.store_command == "fingerprint":
        return _fingerprint(args)
    try:
        if args.store_command == "create":
            fields = stores.create_sidecar(args.primary, args.purpose)
            where = stores.sidecar_path(args.primary)
        elif args.store_command == "adopt":
            fields = stores.adopt_sidecar(args.primary, args.purpose)
            where = stores.sidecar_path(args.primary)
        elif args.store_command == "copy":
            fields = stores.copy_store(
                args.src, args.dst, args.purpose, runs_root=args.runs_root
            )
            where = stores.sidecar_path(args.dst)
        else:
            fields = stores.move_store(args.src, args.dst, runs_root=args.runs_root)
            where = stores.sidecar_path(args.dst)
    except RuntimeError as e:
        print(f"!! {e}", file=sys.stderr)
        return 1
    print(f">> {args.store_command}: {where}")
    print(json.dumps(fields, indent=2, sort_keys=True))
    return 0


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

    store = sub.add_parser("store", help="a datastore and its sidecar")
    store_sub = store.add_subparsers(dest="store_command", required=True)
    runs_root = dict(
        default=DEFAULT_ROOT,
        metavar="DIR",
        help=f"the runs root, default {DEFAULT_ROOT}",
    )
    show = store_sub.add_parser(
        "show", help="the sidecar and the runs naming the store"
    )
    show.add_argument("primary", metavar="PRIMARY")
    show.add_argument("--runs-root", **runs_root)
    create = store_sub.add_parser(
        "create", help="a registry sidecar for a store with none"
    )
    create.add_argument("primary", metavar="PRIMARY")
    create.add_argument("--purpose", required=True, metavar="TEXT")
    adopt = store_sub.add_parser("adopt", help="upgrade a legacy sidecar in place")
    adopt.add_argument("primary", metavar="PRIMARY")
    adopt.add_argument("--purpose", default=None, metavar="TEXT")
    copier = store_sub.add_parser("copy", help="copy a closed store and its sidecar")
    copier.add_argument("src", metavar="SRC")
    copier.add_argument("dst", metavar="DST")
    copier.add_argument("--purpose", required=True, metavar="TEXT")
    copier.add_argument("--runs-root", **runs_root)
    mover = store_sub.add_parser("move", help="move a closed store and its sidecar")
    mover.add_argument("src", metavar="SRC")
    mover.add_argument("dst", metavar="DST")
    mover.add_argument("--runs-root", **runs_root)
    printer = store_sub.add_parser(
        "fingerprint",
        help="the content fingerprint of a closed store, compared with the recorded one",
    )
    printer.add_argument("primary", metavar="PRIMARY")
    printer.add_argument(
        "--write",
        action="store_true",
        help="record it in the sidecar, replacing only its fingerprint field",
    )
    printer.add_argument(
        "--listing",
        default=None,
        metavar="PATH",
        help="also write the full listing to this new file, outside the store's directory",
    )
    printer.add_argument("--runs-root", **runs_root)

    args = parser.parse_args(argv)
    if args.command == "store":
        return _store(args)

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
