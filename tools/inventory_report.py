"""
The datastore inventory report: ``main.py --inventory`` (store-fingerprint prompt 03).

``format_inventory_report(inventory, db_name, verbose=False)`` renders the structured inventory
that ``Datastore.store_inventory.read_inventory`` returns. It reads nothing from a store itself:
whatever it prints is in the ``StoreInventory`` it is given, which was read read-only, with no Ray.

**What it shows.** The report opens with the store's total number of named problems ("none" when
there are none). Then, category by category, each class has:

- a header: its record count, the validated / unvalidated split where the class has the flag,
  the sum of its records' ``value_count`` and the value table it counts where it has one, and the
  timestamp range;
- its tag sets, each with its labels and its record count;
- its records, grouped under their tag set, each rendered by its **physical labels**: a float leaf
  as a number, an integer or string as itself, and a parent reference by the parent's own resolved
  key, never by its digest. Without ``verbose`` at most five records per tag set are shown, and the
  number not shown is given;
- its problems, **every one, never truncated**.

The ``*Value`` tables are not a category: each is reported on its parent's header. The
``store_tag`` class lists every label, whatever ``verbose`` is, and marks a label that no record
carries (tags are created on ``object_get``, so a store can hold one).

**Floats.** A key's float leaves are ``float.hex`` strings (decision D1). Which leaves are floats
is learned from the schema -- ``build_schema``'s column types -- and never from the shape of a
string, so a label that looks like a hex float is printed as the string it is. A float leaf is
rendered as ``float.fromhex(value)``: to six significant figures, or with ``verbose`` as ``repr``,
which round-trips, so two records that differ in the last bit of a leaf render differently.

**What is elided, and why it hides nothing.** A class's records would otherwise repeat, on every
line, fields that every record of the class shares (a store usually holds one background model and
one cosmology). So:

- a key field whose value is the same for **every** record of the class is printed once, on the
  class's "common to every record" line, and left off its record lines;
- a parent is rendered by the fields of its key that vary across **its own** class (and its tags,
  if they vary there), because the others are the same for every parent it could be. If none
  vary -- the class has one distinct key -- it is rendered by its whole key and tags. A lone
  field is printed bare, without its name, where the parent's key has only that field or the field
  is itself a reference: ``z_response=0.1`` for a redshift, ``k=100000`` for an exit time.

Both are decided per class, so every record of a class is rendered through the same fields, and
two records that differ in a key field differ in a field that is shown. Two distinct records of one
class therefore never share a line within a tag set: a record line also shows ``[unvalidated]``
(or ``[validated NULL]``) and its value count, so records that differ only in those differ too.

**Order.** Records are ordered by their resolved physical values, numerically where they are
numbers, never by digest. Fields are in the order of the class's key, as its factory's
``inventory_records`` declares it.
"""

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import sqlalchemy as sqla

from Datastore.store_inventory import (
    COSMOLOGY,
    ClassInventory,
    Record,
    StoreInventory,
    cosmology_classes,
)

# categories are reported in this order, as before this prompt, less the value tables, which are
# now reported on their parents' headers. A class of the inventory that no category names is
# reported under "Other classes", so that nothing the inventory holds can go unprinted
INVENTORY_CATEGORIES: List[Tuple[str, List[str]]] = [
    ("Datastore metadata", ["version", "store_tag"]),
    ("Cosmology models", ["LambdaCDM", "QCD_Cosmology", "BackgroundModel"]),
    (
        "Grid definitions (wavenumbers, redshifts, tolerances)",
        ["wavenumber", "redshift", "wavenumber_exit_time", "tolerance"],
    ),
    (
        "Solvers and policies",
        ["IntegrationSolver", "GkSourcePolicy", "QuadSourcePolicy"],
    ),
    (
        "Compute targets",
        [
            "TkNumericIntegration",
            "TkWKBIntegration",
            "QuadSource",
            "GkNumericIntegration",
            "GkWKBIntegration",
            "GkSource",
            "GkSourcePolicyData",
            "QuadSourceIntegral",
            "OneLoopIntegral",
        ],
    ),
]

# the value table each class's value_count counts: the tables its factory's inventory_records
# names in ``values=``. Used only to name the table on the header
VALUE_TABLES: Dict[str, str] = {
    "BackgroundModel": "BackgroundModelValue",
    "TkNumericIntegration": "TkNumericValue",
    "TkWKBIntegration": "TkWKBValue",
    "QuadSource": "QuadSourceValue",
    "GkNumericIntegration": "GkNumericValue",
    "GkWKBIntegration": "GkWKBValue",
    "GkSource": "GkSourceValue",
}

# at most this many records per tag set without verbose
MAX_SHOWN = 5

_CLASS_INDENT = "      "
_DETAIL_INDENT = "         "
_RECORD_INDENT = "            "

# a string leaf containing one of these is quoted, so that a rendered line cannot be read two ways
_QUOTE_IF = set(",={}[]'\"|")


def _format_timestamp(value: Optional[datetime]) -> str:
    if value is None:
        return "?"
    return value.strftime("%Y-%m-%d %H:%M")


def _format_range(
    earliest: Optional[datetime], latest: Optional[datetime]
) -> Optional[str]:
    if earliest is None and latest is None:
        return None
    return f"{_format_timestamp(earliest)} – {_format_timestamp(latest)}"


def _plural(n: int, word: str) -> str:
    if n == 1:
        return f"{n:,} {word}"
    return f"{n:,} {word}" + ("es" if word.endswith("s") else "s")


def _schema_tables() -> Mapping[str, sqla.Table]:
    from Datastore.SQL.Datastore import _factories
    from Datastore.SQL.schema import build_schema

    return build_schema(sqla.MetaData(), _factories).tables


class _Renderer:
    """Renders records of one ``StoreInventory`` by their physical labels."""

    def __init__(self, inventory: StoreInventory, verbose: bool):
        self.inventory = inventory
        self.verbose = verbose
        self.tables = _schema_tables()
        self.types = cosmology_classes()
        self._index: Dict[str, Dict[str, List[Record]]] = {}
        self._varying: Dict[str, Tuple[Tuple[str, ...], bool]] = {}
        self._float_fields: Dict[str, frozenset] = {}
        self._ref_text: Dict[Tuple[str, str], str] = {}
        self._ref_sort: Dict[Tuple[str, str], tuple] = {}

    # --- what the schema says ----------------------------------------------------------------

    def float_fields(self, name: str) -> frozenset:
        """The columns of ``name``'s table that the schema types as floats."""
        if name not in self._float_fields:
            table = self.tables.get(name)
            self._float_fields[name] = frozenset(
                ()
                if table is None
                else (c.name for c in table.columns if isinstance(c.type, sqla.Float))
            )
        return self._float_fields[name]

    def has_validated(self, name: str) -> bool:
        table = self.tables.get(name)
        return table is not None and "validated" in table.c

    # --- the classes -------------------------------------------------------------------------

    def cls(self, name: str) -> ClassInventory:
        return self.inventory[name]

    def fields(self, name: str) -> Tuple[str, ...]:
        records = self.cls(name).records
        return tuple(records[0].key) if len(records) > 0 else ()

    def varying(self, name: str) -> Tuple[Tuple[str, ...], bool]:
        """The key fields whose value is not the same for every record of ``name``, in key order,
        and whether the tag sets differ between its records."""
        if name not in self._varying:
            records = self.cls(name).records
            fields = tuple(
                f for f in self.fields(name) if len({r.key[f] for r in records}) > 1
            )
            tags = self.cls(name).tagged and len({r.tags for r in records}) > 1
            self._varying[name] = (fields, tags)
        return self._varying[name]

    def parent_class(self, name: str, record: Record, field: str) -> Optional[str]:
        of = self.cls(name).parents.get(field)
        if of == COSMOLOGY:
            return self.types.get(record.key.get("cosmology_type"))
        return of

    def index(self, name: str) -> Dict[str, List[Record]]:
        """reference digest -> the records of ``name`` it names."""
        if name not in self._index:
            out: Dict[str, List[Record]] = {}
            for record in self.cls(name).records:
                out.setdefault(self.inventory.digest_of(name, record), []).append(
                    record
                )
            self._index[name] = out
        return self._index[name]

    # --- leaves ------------------------------------------------------------------------------

    def _float(self, value: Any) -> Optional[float]:
        return None if value is None else float.fromhex(value)

    def leaf_text(self, name: str, field: str, value: Any) -> str:
        if field in self.float_fields(name) and value is not None:
            x = self._float(value)
            return repr(x) if self.verbose else f"{x:.6g}"
        if isinstance(value, str):
            if value == "" or value != value.strip() or _QUOTE_IF & set(value):
                return repr(value)
            return value
        return str(value)

    def leaf_sort(self, name: str, field: str, value: Any) -> tuple:
        if value is None:
            return (0,)
        if field in self.float_fields(name):
            return (1, self._float(value))
        if isinstance(value, (bool, int)):
            return (1, int(value))
        return (2, str(value))

    # --- parents -----------------------------------------------------------------------------

    def _ref_fields(self, name: str, record: Record) -> Tuple[Tuple[str, ...], bool]:
        fields, tags = self.varying(name)
        if len(fields) == 0 and not tags:
            return tuple(record.key), self.cls(name).tagged and len(record.tags) > 0
        return fields, tags

    def ref_text(self, of: Optional[str], reference: str) -> str:
        """A parent, by the fields of its resolved key that vary across its class."""
        if of is None or of not in self.inventory.classes:
            return "<unresolved reference>"
        cache = (of, reference)
        if cache not in self._ref_text:
            found = self.index(of).get(reference, [])
            if len(found) == 0:
                self._ref_text[cache] = f"<unresolved {of} reference>"
            else:
                record = found[0]
                fields, tags = self._ref_fields(of, record)
                parts = [f"{f}={self.value_text(of, record, f)}" for f in fields]
                if tags:
                    parts.append(f"tags=[{', '.join(record.tags)}]")
                # a lone field is printed bare where its name adds nothing: the parent's whole
                # key is that one field (a redshift, a wavenumber, a tolerance), or the field is
                # itself a reference (an exit time that differs only in its k). Decided per class
                if (
                    len(parts) == 1
                    and not tags
                    and (len(record.key) == 1 or fields[0] in self.cls(of).parents)
                ):
                    self._ref_text[cache] = self.value_text(of, record, fields[0])
                else:
                    self._ref_text[cache] = "{" + ", ".join(parts) + "}"
        return self._ref_text[cache]

    def ref_sort(self, of: Optional[str], reference: str) -> tuple:
        if of is None or of not in self.inventory.classes:
            return (9,)
        cache = (of, reference)
        if cache not in self._ref_sort:
            found = self.index(of).get(reference, [])
            if len(found) == 0:
                self._ref_sort[cache] = (9,)
            else:
                record = found[0]
                fields, tags = self._ref_fields(of, record)
                self._ref_sort[cache] = (
                    (3,)
                    + tuple(self.value_sort(of, record, f) for f in fields)
                    + ((record.tags,) if tags else ())
                )
        return self._ref_sort[cache]

    # --- one field of one record -------------------------------------------------------------

    def value_text(self, name: str, record: Record, field: str) -> str:
        value = record.key[field]
        if field in self.cls(name).parents:
            return self.ref_text(self.parent_class(name, record, field), value)
        return self.leaf_text(name, field, value)

    def value_sort(self, name: str, record: Record, field: str) -> tuple:
        value = record.key[field]
        if field in self.cls(name).parents:
            return self.ref_sort(self.parent_class(name, record, field), value)
        return self.leaf_sort(name, field, value)

    # --- a record's line ---------------------------------------------------------------------

    def line_fields(self, name: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """(the fields a record line shows, the fields on the class's common line)."""
        fields, _ = self.varying(name)
        if len(fields) == 0:
            return self.fields(name), ()
        return fields, tuple(f for f in self.fields(name) if f not in fields)

    def record_line(self, name: str, record: Record, fields: Sequence[str]) -> str:
        text = ", ".join(f"{f}={self.value_text(name, record, f)}" for f in fields)
        if self.has_validated(name):
            if record.validated is False:
                text += "  [unvalidated]"
            elif record.validated is None:
                text += "  [validated NULL]"
        if record.value_count is not None:
            text += f"  | values: {record.value_count:,}"
        return text

    def record_sort(self, name: str, record: Record, fields: Sequence[str]) -> tuple:
        return tuple(self.value_sort(name, record, f) for f in fields) + (
            record.canonical_json(),
        )


def _class_lines(
    renderer: _Renderer, name: str, carried: frozenset, verbose: bool
) -> List[str]:
    cls = renderer.cls(name)
    records = cls.records
    where = "replicated" if cls.replicated else "sharded"

    header = f"{_CLASS_INDENT}@@ {name} ({where}): {_plural(cls.count, 'record')}"
    parts = []
    if renderer.has_validated(name):
        validated = sum(1 for r in records if r.validated is True)
        unvalidated = sum(1 for r in records if r.validated is False)
        null = sum(1 for r in records if r.validated is None)
        split = f"{validated:,} validated, {unvalidated:,} unvalidated"
        if null > 0:
            split += f", {null:,} validated NULL"
        parts.append(split)
    if name in VALUE_TABLES:
        total = sum(r.value_count or 0 for r in records)
        parts.append(f"{_plural(total, VALUE_TABLES[name] + ' row')}")
    if len(parts) > 0:
        header += ": " + "; ".join(parts)
    rng = _format_range(cls.earliest_timestamp, cls.latest_timestamp)
    if rng is not None:
        header += f" | {rng}"
    if cls.count == 0:
        header += " (empty)"
    lines = [header]

    if cls.count > 0:
        fields, common = renderer.line_fields(name)
        if len(common) > 0:
            lines.append(
                f"{_DETAIL_INDENT}common to every record: "
                + ", ".join(
                    f"{f}={renderer.value_text(name, records[0], f)}" for f in common
                )
            )

        groups: Dict[Tuple[str, ...], List[Record]] = {}
        for record in records:
            groups.setdefault(record.tags, []).append(record)
        ordered = sorted(groups.items())

        for number, (tags, members) in enumerate(ordered, start=1):
            if cls.tagged:
                labels = ", ".join(tags) if len(tags) > 0 else "(no tags)"
                lines.append(
                    f"{_DETAIL_INDENT}tag set {number} of {len(ordered)} "
                    f"({_plural(len(members), 'record')}): {labels}"
                )
            members = sorted(
                members, key=lambda r: renderer.record_sort(name, r, fields)
            )
            complete = verbose or name == "store_tag"
            shown = members if complete else members[:MAX_SHOWN]
            for record in shown:
                line = renderer.record_line(name, record, fields)
                if name == "store_tag" and record.key.get("label") not in carried:
                    line += "  (carried by no record)"
                lines.append(f"{_RECORD_INDENT}- {line}")
            if len(members) > len(shown):
                lines.append(
                    f"{_RECORD_INDENT}... and {len(members) - len(shown):,} more"
                )

    if len(cls.problems) > 0:
        lines.append(f"{_DETAIL_INDENT}{_plural(len(cls.problems), 'problem')}:")
        for problem in cls.problems:
            lines.append(f"{_RECORD_INDENT}!! {problem}")

    return lines


def format_inventory_report(
    inventory: StoreInventory, db_name: Any, verbose: bool = False
) -> str:
    """
    Render ``inventory`` (what ``read_inventory`` returns) as the text of the inventory report.
    ``db_name`` is printed in the title as given. With ``verbose``, every record is listed and
    floats are printed as ``repr``; otherwise at most five records per tag set, and floats to six
    significant figures. Every problem is printed either way.
    """
    renderer = _Renderer(inventory, verbose)
    problems = inventory.problems
    total = sum(c.count for c in inventory.classes.values())
    carried = frozenset(
        tag for c in inventory.classes.values() for r in c.records for tag in r.tags
    )

    lines = [f"== Datastore inventory: {db_name} =="]
    lines.append(
        "   problems: none"
        if len(problems) == 0
        else f"   problems: {len(problems):,}, each listed in full under its class"
    )
    shard_list = ", ".join(f"#{s}" for s in inventory.shards)
    lines.append(
        f"   read-only; {_plural(len(inventory.shards), 'shard')} ({shard_list}); "
        f"{_plural(total, 'record')} in {_plural(len(inventory.classes), 'class')}"
    )

    named = {n for _, names in INVENTORY_CATEGORIES for n in names}
    categories = [
        (title, [n for n in names if n in inventory.classes])
        for title, names in INVENTORY_CATEGORIES
    ]
    others = [n for n in inventory.classes if n not in named]
    if len(others) > 0:
        categories.append(("Other classes", others))

    for title, names in categories:
        lines.append("")
        lines.append(f"   -- {title}")
        for name in names:
            lines.extend(_class_lines(renderer, name, carried, verbose))

    return "\n".join(lines)
