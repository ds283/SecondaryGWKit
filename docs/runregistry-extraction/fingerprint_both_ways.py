"""
The fingerprint both ways (runregistry-extraction prompt 04, W1; README §6.2 C3; 08 runs it again).

For each primary given, read-only and with no Ray, this takes the store's fingerprint twice from one
inventory and compares the two:

- **SGK's**: ``RunRegistry.stores.fingerprint_of(inventory)``, the fingerprint the run registry
  records today;
- **the package's**: ``datastorekit.store_fingerprint.fingerprint_of(inventory, FINGERPRINT_FORMAT)``,
  with SGK's format passed from SGK's own constant.

The inventory is ``datastorekit.store_inventory.read_inventory(primary, factories)``, with
``factories`` from ``config.datastore``, as ``RunRegistry.stores.fingerprint_store`` reads it.

The package's fingerprint must equal SGK's in every field but one, ``encoding_format``, which the
package adds beside the digest and SGK's code does not have (DatastoreKit ``v0.4.0``'s SGK
addendum, items 3, 4 and 7). The checks, each printed with its result:

1. the package's ``encoding_format`` is ``ENCODING_FORMAT``, and is ``1``;
2. SGK's fingerprint has no ``encoding_format``;
3. with ``encoding_format`` removed from the package's, the two are equal as dictionaries, every
   differing path printed with both values; and in the same key order, at the top level and over
   the classes (``dict`` equality ignores order; ``tag_sets`` are lists, so ``==`` compares theirs);
4. ``fingerprint_format`` is 6 in both, and ``taken`` is ``None`` in both;
5. the overall digest, printed, is ``EXPECTED_DIGEST`` in both;
6. the listings: SGK's ``listing_lines(inventory, sgk)`` and the package's
   ``listing_lines(inventory, pkg)`` are equal line for line but the first, and the package's first
   line is SGK's with `` encoding_format 1`` appended;
7. the comparisons: the package's ``compare_fingerprints(pkg, pkg)`` and SGK's
   ``compare_fingerprints(sgk, sgk)`` are empty, and the package's ``compare_fingerprints(sgk,
   pkg)`` is exactly one entry, of kind ``encoding``
   (``[00-recorded-fingerprints-lack-the-encoding-format]``, measured);
8. against the sidecar: SGK's ``compare_fingerprints`` of the sidecar's recorded fingerprint with
   SGK's fresh one is empty.

It prints one block per store and a summary line, and exits 0 if and only if every check holds on
every store; 1 otherwise; 2 on a usage error. It writes nothing, anywhere, and changes nothing in
``RunRegistry``, which it only imports. It is a verification script, not a test (README §5 rule 10).

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/runregistry-extraction/fingerprint_both_ways.py \\
        var/datastores/runregistry-extraction-04-reference.sqlite ...
"""

import argparse
import sys
from pathlib import Path

EXPECTED_DIGEST = "39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2"
EXPECTED_FORMAT = 6
EXPECTED_ENCODING = 1


def _tags_label(entry) -> str:
    tags = entry.get("tags") if isinstance(entry, dict) else None
    return "[" + ", ".join(tags) + "]" if isinstance(tags, list) else "?"


def differences(a, b, path="") -> list:
    """Every path at which ``a`` and ``b`` differ, as ``(path, a's value, b's value)``.

    A path names the class (``classes.<name>``), the tag set by its tags
    (``tag_sets[<i>] tags [<tags>]``) and the field. A key present on one side only is reported
    with ``<absent>`` on the other.
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


def first_order_difference(a, b):
    """The first position at which two key lists differ, or ``None`` if they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        i = min(len(a), len(b))
        return i, (a[i] if i < len(a) else "<end>"), (b[i] if i < len(b) else "<end>")
    return None


def check_store(primary: Path, report) -> bool:
    from config.datastore import factories
    from datastorekit import store_fingerprint as package
    from datastorekit.store_inventory import read_inventory
    from RunRegistry import stores as sgk_stores

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

    inventory = read_inventory(str(primary), factories)
    sgk = sgk_stores.fingerprint_of(inventory)
    pkg = package.fingerprint_of(inventory, sgk_stores.FINGERPRINT_FORMAT)
    report(
        f"  SGK format {sgk.get('fingerprint_format')}, digest {sgk.get('digest')}, "
        f"{len(sgk.get('classes') or {})} classes"
    )
    report(
        f"  package format {pkg.get('fingerprint_format')}, encoding "
        f"{pkg.get('encoding_format')}, digest {pkg.get('digest')}, "
        f"{len(pkg.get('classes') or {})} classes"
    )

    # 1, 2. the encoding format
    check(
        "package encoding_format == ENCODING_FORMAT",
        pkg.get("encoding_format") == package.ENCODING_FORMAT,
        f"{pkg.get('encoding_format')!r} and {package.ENCODING_FORMAT!r}",
    )
    check(
        f"package encoding_format == {EXPECTED_ENCODING}",
        pkg.get("encoding_format") == EXPECTED_ENCODING,
        repr(pkg.get("encoding_format")),
    )
    check(
        "'encoding_format' not in SGK's fingerprint",
        "encoding_format" not in sgk,
        f"SGK's keys {list(sgk)}",
    )

    # 3. equal but for encoding_format, as dictionaries and in order
    stripped = {k: v for k, v in pkg.items() if k != "encoding_format"}
    diffs = differences(stripped, sgk)
    check(
        "package without encoding_format == SGK's, as dictionaries",
        stripped == sgk and not diffs,
        f"{len(diffs)} differing path(s)",
    )
    for path, a, b in diffs:
        report(f"      {path}: package {a!r}, SGK {b!r}")
    for label, a, b in (
        ("the keys", list(stripped), list(sgk)),
        (
            "the classes",
            list(stripped.get("classes") or {}),
            list(sgk.get("classes") or {}),
        ),
    ):
        where = first_order_difference(a, b)
        check(
            f"{label} in the same order",
            where is None,
            (
                f"{len(a)} in order"
                if where is None
                else f"first differ at position {where[0]}: package {where[1]!r}, SGK {where[2]!r}"
            ),
        )

    # 4. the client's format, and taken
    for who, fp in (("SGK", sgk), ("package", pkg)):
        check(
            f"{who} fingerprint_format == {EXPECTED_FORMAT}",
            fp.get("fingerprint_format") == EXPECTED_FORMAT,
            repr(fp.get("fingerprint_format")),
        )
        check(f"{who} taken is None", fp.get("taken") is None, repr(fp.get("taken")))

    # 5. the overall digest
    for who, fp in (("SGK", sgk), ("package", pkg)):
        check(
            f"{who} digest == {EXPECTED_DIGEST[:8]}…",
            fp.get("digest") == EXPECTED_DIGEST,
            str(fp.get("digest")),
        )

    # 6. the listings
    sgk_lines = list(sgk_stores.listing_lines(inventory, sgk))
    pkg_lines = list(package.listing_lines(inventory, pkg))
    check(
        "listings have the same length",
        len(sgk_lines) == len(pkg_lines),
        f"SGK {len(sgk_lines)}, package {len(pkg_lines)}",
    )
    rest = [i for i, (x, y) in enumerate(zip(sgk_lines, pkg_lines)) if i > 0 and x != y]
    check(
        "listings equal line for line but the first",
        not rest and len(sgk_lines) == len(pkg_lines),
        f"{len(rest)} differing line(s)"
        + (f", first at line {rest[0] + 1}" if rest else ""),
    )
    expected_first = f"{sgk_lines[0]} encoding_format {EXPECTED_ENCODING}"
    check(
        "package's first line is SGK's + ' encoding_format 1'",
        pkg_lines[:1] == [expected_first],
        f"SGK {sgk_lines[0]!r}; package {pkg_lines[0]!r}",
    )

    # 7. the comparisons, as the addendum states them
    pp = package.compare_fingerprints(pkg, pkg)
    check("package compare_fingerprints(pkg, pkg) is empty", pp == [], f"{len(pp)}")
    ss = sgk_stores.compare_fingerprints(sgk, sgk)
    check("SGK compare_fingerprints(sgk, sgk) is empty", ss == [], f"{len(ss)}")
    sp = package.compare_fingerprints(sgk, pkg)
    check(
        "package compare_fingerprints(sgk, pkg) is one entry, kind 'encoding'",
        len(sp) == 1 and sp[0].get("kind") == "encoding",
        "; ".join(
            f"{e.get('kind')}: recorded {e.get('recorded')!r}, current {e.get('current')!r}"
            for e in sp
        ),
    )

    # 8. against the sidecar
    reading = sgk_stores.read_sidecar(str(primary))
    recorded = (reading.fields or {}).get("fingerprint") if reading.fields else None
    if check(
        "the sidecar records a fingerprint",
        reading.kind == "registry" and isinstance(recorded, dict),
        f"sidecar kind {reading.kind}",
    ):
        rs = sgk_stores.compare_fingerprints(recorded, sgk)
        taken = recorded.get("taken") or {}
        check(
            "SGK compare_fingerprints(sidecar's, sgk) is empty",
            rs == [],
            f"{len(rs)} difference(s); recorded {str(recorded.get('digest'))[:8]}…, taken "
            f"{taken.get('when')} by {taken.get('run_id')}",
        )
        for e in rs:
            report(f"      {e.get('text')}")

    ok = all(results)
    report(
        f"  => {'every check holds' if ok else 'FAILED'} "
        f"({sum(results)} of {len(results)} checks hold)"
    )
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare SGK's and DatastoreKit's fingerprints of each store (W1)."
    )
    parser.add_argument("primary", nargs="+", help="a store's primary .sqlite file")
    args = parser.parse_args(argv)

    failed = []
    for i, name in enumerate(args.primary):
        if i:
            print()
        if not check_store(Path(name), print):
            failed.append(name)
    print()
    n = len(args.primary)
    if failed:
        print(f"summary: {n} store(s); FAILED on {len(failed)}: {', '.join(failed)}")
        return 1
    print(
        f"summary: {n} store(s); every check holds on every store: the package's fingerprint "
        f"equals SGK's in every field but encoding_format"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
