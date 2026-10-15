"""
The fingerprint both ways, across stores (runregistry-extraction prompt 06, for 08; README §4).

04's form of this script (W1, at `ec61569`) took one store's fingerprint twice from one inventory,
through this repository's own `fingerprint_of` and through DatastoreKit's, and compared the two.
Prompt 06 deleted the first. So the comparison is now made across stores: each store's fingerprint,
taken through the fingerprinter this repository hands RunRegistryKit, against the fingerprints this
repository's own code recorded for earlier stores in their run directories. 08 uses it to carry
W1's claim to the tree where only the package's implementation is left.

For each primary given, read-only and with no Ray, the store's fingerprint is
``config.fingerprint.fingerprinter.take(primary, None)``. Each ``--against`` is a run directory
whose ``status.json`` carries a ``fingerprint``, or that ``status.json``, as
``docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py`` reads it. The checks, each printed
with its result:

- for each store: its ``encoding_format`` is ``datastorekit.store_fingerprint.ENCODING_FORMAT``,
  and its overall digest is ``--expect-digest``;
- against the store's sidecar, which the package wrote: ``compare_fingerprints(sidecar's,
  current)`` is empty;
- for each store and each recorded fingerprint:
  - the recorded fingerprint lacks ``encoding_format``, and its overall digest is
    ``--expect-digest``;
  - **field by field**, with ``encoding_format`` and ``taken`` set aside on both sides, the two are
    equal: the same keys, ``fingerprint_format``, the overall ``digest``, ``problems``, and for every
    class its ``count``, its ``digest`` and every tag set's ``tags``, ``count`` and ``digest``, the
    tag sets in the same order. Every differing path is printed with both values;
  - the package's ``compare_fingerprints(recorded, current)`` is exactly one entry, of kind
    ``encoding`` (``[00-recorded-fingerprints-lack-the-encoding-format]``, measured, not a failure).

**Why field by field.** A fingerprint this repository's own code recorded has no
``encoding_format``. The package's ``compare_fingerprints`` compares ``fingerprint_format``, then
``encoding_format``, and when either differs it gives one entry for it and compares nothing else.
So against such a record it says nothing about the content, and this script compares the content
itself, with the one field the record cannot have set aside. **Key order is not compared.** Every
record, a run's ``status.json`` and a sidecar alike, is written with ``sort_keys=True``, so neither
the order of a fingerprint's keys nor the order of its classes survives in it; a fresh fingerprint
lists its classes in registry order. The order of the tag sets, a list, is recorded, and is
compared.

It prints one block per store and a summary line, and exits 0 if and only if every check holds for
every store and every recorded fingerprint; 1 otherwise; 2 on a usage error. It writes nothing,
anywhere. It imports ``runregistrykit``, ``datastorekit`` and ``config.fingerprint``. It is a
verification script, not a test (README §5 rule 10).

Run from the repository root::

    ./venv/bin/python docs/runregistry-extraction/fingerprint_both_ways.py \\
        var/datastores/<store>.sqlite [...] --against var/runs/<run id> [...]

``--expect-digest`` defaults to the reference fingerprint's digest, ``39809dca…``, which is what 08
uses; another digest exercises the script on a fixture store.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REFERENCE_DIGEST = "39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2"

# set aside on both sides of the field-by-field comparison: the one the record cannot have, and
# the one that is never content
SET_ASIDE = ("encoding_format", "taken")


def _tags_label(entry) -> str:
    tags = entry.get("tags") if isinstance(entry, dict) else None
    return "[" + ", ".join(tags) + "]" if isinstance(tags, list) else "?"


def differences(a, b, path="") -> list:
    """Every path at which ``a`` and ``b`` differ, as ``(path, a's value, b's value)``.

    A path names the class (``classes.<name>``), the tag set by its tags
    (``tag_sets[<i>] tags [<tags>]``) and the field. A key present on one side only is reported
    with ``<absent>`` on the other. Dictionaries are compared by key, not by order (the module
    docstring); lists position by position.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        keys = list(a) + [k for k in b if k not in a]
        for k in keys:
            sub = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append((sub, "<absent>", b[k]))
            elif k not in b:
                out.append((sub, a[k], "<absent>"))
            else:
                out.extend(differences(a[k], b[k], sub))
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = []
        if len(a) != len(b):
            out.append((f"{path} (length)", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            label = f"{path}[{i}]"
            if path.endswith("tag_sets"):
                ta, tb = _tags_label(x), _tags_label(y)
                label += f" tags {ta}" if ta == tb else f" tags {ta} / {tb}"
            out.extend(differences(x, y, label))
        return out
    if a != b or type(a) is not type(b):
        return [(path, a, b)]
    return []


def set_aside(fingerprint) -> dict:
    return {k: v for k, v in fingerprint.items() if k not in SET_ASIDE}


def recorded(path: Path):
    """``(status.json, its fingerprint)`` for a run directory or a ``status.json``."""
    status = path / "status.json" if path.is_dir() else path
    data = json.loads(status.read_text())
    if not isinstance(data.get("fingerprint"), dict):
        raise ValueError(f"{status} records no fingerprint")
    return status, data["fingerprint"]


def check_store(primary: Path, records, expect_digest, report) -> bool:
    from datastorekit.store_fingerprint import ENCODING_FORMAT, compare_fingerprints
    from runregistrykit.stores import read_sidecar

    from config.fingerprint import fingerprinter

    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        report(
            f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else "")
        )
        return ok

    report(f"== {primary}")
    if not primary.is_file():
        check("the primary exists", False, "no such file")
        return False

    current, _ = fingerprinter.take(primary, None)
    report(
        f"  current: format {current.get('fingerprint_format')}, encoding "
        f"{current.get('encoding_format')}, digest {current.get('digest')}, "
        f"{len(current.get('classes') or {})} classes"
    )

    # the store
    check(
        "current encoding_format == ENCODING_FORMAT",
        current.get("encoding_format") == ENCODING_FORMAT,
        f"{current.get('encoding_format')!r} and {ENCODING_FORMAT!r}",
    )
    check(
        f"current digest == {expect_digest[:8]}…",
        current.get("digest") == expect_digest,
        str(current.get("digest")),
    )

    # against the sidecar, which the package wrote
    reading = read_sidecar(str(primary), client_root=str(REPO_ROOT))
    sidecar = (reading.fields or {}).get("fingerprint") if reading.fields else None
    if check(
        "the sidecar records a fingerprint",
        reading.kind == "registry" and isinstance(sidecar, dict),
        f"sidecar kind {reading.kind}",
    ):
        rs = compare_fingerprints(sidecar, current)
        taken = sidecar.get("taken") or {}
        check(
            "compare_fingerprints(sidecar's, current) is empty",
            rs == [],
            f"{len(rs)} entr{'y' if len(rs) == 1 else 'ies'}; sidecar's digest "
            f"{str(sidecar.get('digest'))[:8]}…, encoding {sidecar.get('encoding_format')!r}, "
            f"taken {taken.get('when')}",
        )
        for e in rs:
            report(f"      {e.get('kind')}: {e.get('text')}")

    # each recorded fingerprint
    for status, fp in records:
        taken = fp.get("taken") or {}
        report(
            f"  -- against {status}: format {fp.get('fingerprint_format')}, digest "
            f"{fp.get('digest')}, {len(fp.get('classes') or {})} classes; taken by "
            f"{taken.get('run_id')} at {taken.get('git_head')}"
        )
        check(
            "'encoding_format' not in the recorded fingerprint",
            "encoding_format" not in fp,
            f"recorded keys {sorted(fp)}",
        )
        check(
            f"recorded digest == {expect_digest[:8]}…",
            fp.get("digest") == expect_digest,
            str(fp.get("digest")),
        )
        diffs = differences(set_aside(current), set_aside(fp))
        check(
            "field by field, encoding_format and taken set aside, current == recorded",
            not diffs,
            f"{len(diffs)} differing path(s)",
        )
        for path, a, b in diffs:
            report(f"      {path}: current {a!r}, recorded {b!r}")
        rc = compare_fingerprints(fp, current)
        check(
            "compare_fingerprints(recorded, current) is one entry, kind 'encoding'",
            len(rc) == 1 and rc[0].get("kind") == "encoding",
            "; ".join(
                f"{e.get('kind')}: recorded {e.get('recorded')!r}, current "
                f"{e.get('current')!r}"
                for e in rc
            )
            or "no entries",
        )

    ok = all(results)
    report(
        f"  => {'every check holds' if ok else 'FAILED'} "
        f"({sum(results)} of {len(results)} checks hold)"
    )
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare each store's fingerprint, taken through config.fingerprint, with "
            "fingerprints recorded in run directories, field by field (README §4)."
        )
    )
    parser.add_argument("primary", nargs="+", help="a store's primary .sqlite file")
    parser.add_argument(
        "--against",
        nargs="+",
        required=True,
        metavar="RUN_DIR",
        help="a run directory whose status.json carries a fingerprint, or that status.json",
    )
    parser.add_argument(
        "--expect-digest",
        default=REFERENCE_DIGEST,
        help="the overall digest every fingerprint must have (default: the reference's, "
        f"{REFERENCE_DIGEST[:8]}…)",
    )
    args = parser.parse_args(argv)

    try:
        records = [recorded(Path(p)) for p in args.against]
    except (OSError, ValueError) as e:
        print(f"fingerprint_both_ways: {e}", file=sys.stderr)
        return 2

    failed = []
    for i, name in enumerate(args.primary):
        if i:
            print()
        if not check_store(Path(name), records, args.expect_digest, print):
            failed.append(name)
    print()
    n, m = len(args.primary), len(records)
    if failed:
        print(
            f"summary: {n} store(s) against {m} recorded fingerprint(s); FAILED on "
            f"{len(failed)}: {', '.join(failed)}"
        )
        return 1
    print(
        f"summary: {n} store(s) against {m} recorded fingerprint(s); every check holds: each "
        f"store's fingerprint equals every recorded one in every field but encoding_format and "
        f"taken"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
