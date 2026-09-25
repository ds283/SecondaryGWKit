"""`python -m RunRegistry list` — one line per run, newest first.

Run it at the start of a session and before launching anything long. A registry nobody reads is
worse than none, because it looks like coverage.

`python -m RunRegistry store {show,create,adopt,copy,move,fingerprint,retire,amend}` manages a
datastore and its `<stem>.manifest.json` sidecar (`RunRegistry.stores`). `show` is read-only.
`create` and `adopt` write a sidecar and never open the store. `copy` and `move` move the store's
files with `ShardedPool` and carry the sidecar, and refuse a store that any `running` run names,
alive or stale. `fingerprint` reads the closed store read-only, compares its content fingerprint
with the one the sidecar records, and writes it into the sidecar only with `--write`; it refuses a
store a `running` run names, and `--listing` writes the full listing to a new file away from the
store. `retire` deletes a closed store's own files, its shards and then its primary, through
`ShardedPool.delete_store`, and keeps its sidecar as a tombstone that says when, why and what the
store held; it refuses a store a `running` run names, alive or stale, and one whose recorded
fingerprint does not match its content, and `--dry-run` reports what it would do and does nothing.
`amend` replaces or removes one **unknown** field of the sidecar, with a required `--reason`, and
records the field's old value in an `amend` history entry; it is the one way to change an unknown
field, and it never touches a known one. It refuses a store a `running` run names, alive or stale.
None of them initialises Ray. `retire` alone deletes, and only a store's own files; none deletes a
sidecar, a run directory or any other record, and every other command refuses a tombstone.
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


def _print_references(references) -> None:
    if not isinstance(references, dict):
        print(f"  references: {references!r}")
        return
    runs = references.get("runs") or []
    print(f"  runs naming it, under {references.get('runs_root')}: {len(runs)}")
    for run in runs:
        print(
            f"    {run.get('id')}  {(run.get('state') or 'unknown'):9} by "
            f"{' and '.join(run.get('matched_by') or [])}"
        )
    sidecars = references.get("sidecars") or []
    print(
        f"  sidecars naming it, under {references.get('stores_root')}: {len(sidecars)}"
    )
    for entry in sidecars:
        if "unreadable" in entry:
            print(f"    !! {entry.get('sidecar')}: unreadable, {entry['unreadable']}")
        else:
            print(f"    {entry.get('sidecar')}: {', '.join(entry.get('fields') or [])}")
    print(f"  not searched: {references.get('not_searched')}")


def _print_condition(condition) -> None:
    condition = condition if isinstance(condition, dict) else {}
    if condition.get("condition") == "matched":
        print(
            f"  matched: a fresh read-only fingerprint matched the recorded one, digest "
            f"{condition.get('digest')}"
        )
    elif condition.get("condition") == "without":
        print(
            f"  without (--without-fingerprint): the store could not be fingerprinted: "
            f"{condition.get('error_type')}: {condition.get('error')}"
        )
    else:
        print(f"  {condition!r}")


def _print_retirement(retired) -> None:
    """A tombstone's `retired` field, for `store show` and `store retire`."""
    if not isinstance(retired, dict):
        print(f"retirement: {retired!r} (malformed)")
        return
    print("retirement:")
    print(f"  state:     {retired.get('state')}")
    print(
        f"  when:      {retired.get('when')} at {retired.get('git_head')}"
        f"{' (dirty)' if retired.get('git_dirty') else ''}"
    )
    print(f"  reason:    {retired.get('reason')}")
    print(f"  completed: {retired.get('completed') or 'not yet'}")
    print("  fingerprint:")
    _print_condition(retired.get("fingerprint"))
    listed = (
        "the files present when it began; a shard was already missing"
        if retired.get("files_present_only")
        else "every file of the store"
    )
    print(f"  files ({listed}):")
    for path in retired.get("files") or []:
        print(f"    {path}")
    print("  references, when it was retired:")
    _print_references(retired.get("references"))


def _show(args) -> int:
    reading = stores.read_sidecar(args.primary)
    if reading.retired:
        _print_retirement(reading.fields.get("retired"))
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
    # a completed tombstone is not a problem; an incomplete retirement, or a primary that exists
    # again at a retired name, is. Any other sidecar exits 0 whatever it reads as, as before
    return 1 if reading.retired and reading.problems else 0


def _retire(args) -> int:
    try:
        result = stores.retire_store(
            args.primary,
            args.reason,
            runs_root=args.runs_root,
            stores_root=args.stores_root,
            without_fingerprint=args.without_fingerprint,
            dry_run=args.dry_run,
        )
    except RuntimeError as e:
        print(f"!! {e}", file=sys.stderr)
        return 1

    print(f"store:    {result['primary']}")
    print(f"sidecar:  {result['sidecar']}")
    if result["dry_run"]:
        print("dry run:  nothing was written or deleted")
    if result["completing"]:
        print(
            "completing a retirement that was interrupted; its tombstone is kept as written"
        )
    print("references:")
    _print_references(result["references"])
    print("fingerprint check:")
    _print_condition(result["fingerprint"])
    if result["dry_run"]:
        print("files to be deleted:")
        shown = result["to_delete"]
    else:
        print("files deleted:")
        shown = result["deleted"]
    for path in shown:
        print(f"  {path}")
    if not shown:
        print("  none")
    print("tombstone:")
    for line in json.dumps(result["tombstone"], indent=2, sort_keys=True).splitlines():
        print(f"  {line}")
    if not result["dry_run"]:
        print(f">> retired: {result['sidecar']}")
    return 0


def _amend(args) -> int:
    kwargs = {}
    if args.json is not None:
        try:
            kwargs["value"] = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(
                f'!! Cannot amend "{args.primary}": --json {args.json!r} does not parse '
                f"({type(e).__name__}: {e}). Nothing was written",
                file=sys.stderr,
            )
            return 1
    if args.remove:
        kwargs["remove"] = True

    try:
        result = stores.amend_sidecar(
            args.primary, args.field, args.reason, runs_root=args.runs_root, **kwargs
        )
    except RuntimeError as e:
        print(f"!! {e}", file=sys.stderr)
        return 1

    print(f"store:    {result['primary']}")
    print(f"sidecar:  {result['sidecar']}")
    print(f"field:    {result['field']}")
    print(f"before:   {json.dumps(result['before'], sort_keys=True)}")
    print(f"after:    {json.dumps(result['after'], sort_keys=True)}")
    print("history entry:")
    for line in json.dumps(result["entry"], indent=2, sort_keys=True).splitlines():
        print(f"  {line}")
    print(f">> amended: {result['sidecar']}")
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
    if args.store_command == "retire":
        return _retire(args)
    if args.store_command == "amend":
        return _amend(args)
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
    retirer = store_sub.add_parser(
        "retire",
        help="delete a closed store's files, keeping its sidecar as a tombstone",
    )
    retirer.add_argument("primary", metavar="PRIMARY")
    retirer.add_argument(
        "--reason", required=True, metavar="TEXT", help="why the store is retired"
    )
    retirer.add_argument(
        "--without-fingerprint",
        action="store_true",
        help="for a store that cannot be fingerprinted: record the error that prevents it",
    )
    retirer.add_argument(
        "--dry-run",
        action="store_true",
        help="make every check and report what would be done; write and delete nothing",
    )
    retirer.add_argument("--runs-root", **runs_root)
    retirer.add_argument(
        "--stores-root",
        default=stores.DEFAULT_STORES_ROOT,
        metavar="DIR",
        help=f"where sidecars that reference the store are searched for, default "
        f"{stores.DEFAULT_STORES_ROOT}",
    )
    amender = store_sub.add_parser(
        "amend",
        help="replace or remove one unknown field of the sidecar, recording the old value",
    )
    amender.add_argument("primary", metavar="PRIMARY")
    amender.add_argument(
        "--field", required=True, metavar="NAME", help="the field's name"
    )
    amender.add_argument(
        "--json", default=None, metavar="VALUE", help="the field's new value, as JSON"
    )
    amender.add_argument(
        "--remove", action="store_true", help="remove the field instead of replacing it"
    )
    amender.add_argument(
        "--reason", required=True, metavar="TEXT", help="why the field is amended"
    )
    amender.add_argument("--runs-root", **runs_root)

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
