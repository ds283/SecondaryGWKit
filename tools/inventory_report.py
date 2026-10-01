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
  key, never by its digest -- except a reference to a **derived row**, below. Without ``verbose`` at
  most five records per tag set are shown, and the number not shown is given;
- its problems, **every one, never truncated**.

**Derived rows are labels** (prompts/datastore-integrity prompt 09d). A parent reference whose class
is sharded -- the Tk and Gk numeric and WKB integrations, ``QuadSource``, ``GkSource``,
``GkSourcePolicyData`` -- is printed as a label ``<Class>#<n>``, where ``n`` is the row's ordinal in
its own class's verbose listing (tag set by tag set, in physical order, from 1). A row is printed
once, in its own class's section, where its record line carries its own label after its fields
(``  #3``, beyond the first double space, so that a parser of the fields does not read it). A
label depends on no serial. A reference to a replicated class (a model, an exit time, a redshift, a
tolerance, a policy) is rendered from the parent's key, as before. A set of parents (a
``GkSource``'s, prompt 09b) is labelled ``S<n>`` the same way, by its ordinal in physical order, and
a record prints it as ``parents=S<n>``. Each set is printed once, after the records of its class's
section, as ``parent set S<n>: [{z_source=..., numeric=..., wkb=...}, ...]``, its members in order
of their physical values, each field a label or ``none``.

Every label printed is defined in the report. Without ``verbose`` a class shows at most five
records per tag set, so a row that a shown record names may not be among them: each class section
then also lists those rows, one line each, under the prefix ``>``, and the sets that the records
shown and listed name. The listing is closed under reference, which bounds the report: it holds
what the records shown name, and what those name. A reference that cannot be resolved is printed
``<unresolved ... reference>`` or ``<unresolved parent set>``.

Two distinct records still never render as one line: a label is distinct for each distinct row,
and a set label for each distinct set, so no key field is dropped by naming a parent. A summary of
a set, such as its member count and ``z_source`` range, would not keep that.

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
- a replicated parent is rendered by the fields of its key that vary across **its own** class (and
  its tags, if they vary there), because the others are the same for every parent it could be. If
  none vary -- the class has one distinct key -- it is rendered by its whole key and tags. A lone
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
        self._groups: Dict[str, List[Tuple[Tuple[str, ...], List[Record]]]] = {}
        self._ordinals: Dict[
            str, Tuple[Dict[int, int], Dict[str, int], List[Record]]
        ] = {}
        self._labelled: Optional[frozenset] = None
        self._set_numbers: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None
        self._set_digests: Dict[Tuple[str, str], Dict[int, str]] = {}
        self._needed: Optional[
            Tuple[Dict[str, List[int]], Dict[Tuple[str, str], List[int]]]
        ] = None

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
            elif of in self.labelled():
                # a derived row is named by its label, and printed once, in its own section
                self._ref_text[cache] = f"{of}#{self.ordinals(of)[1][reference]}"
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

    # --- sets of parents (prompts/datastore-integrity prompt 09b) ------------------------------

    def _set_members(self, name: str, field: str, value: str) -> Optional[list]:
        """The resolved members of the set whose digest is ``value``, in physical order, each as
        (its sort key, its text); None if the inventory holds no such set."""
        members = self.cls(name).members.get(field, {}).get(value)
        if members is None:
            return None
        classes = self.cls(name).parent_sets[field]
        out = []
        for member in members:
            parts = []
            sort = []
            for (member_field, of), reference in zip(classes.items(), member):
                if reference is None:
                    parts.append(f"{member_field}=none")
                    sort.append((0,))
                else:
                    parts.append(f"{member_field}={self.ref_text(of, reference)}")
                    sort.append(self.ref_sort(of, reference))
            out.append((tuple(sort), "{" + ", ".join(parts) + "}"))
        return sorted(out)

    def members_text(self, name: str, field: str, value: str) -> str:
        """The members of a set, in physical order, each field a label or ``none``."""
        members = self._set_members(name, field, value)
        if members is None:
            return "<unresolved parent set>"
        return "[" + ", ".join(text for _, text in members) + "]"

    def set_text(self, name: str, field: str, value: str) -> str:
        """A set, by its label ``S<n>``; the set itself is printed once, in its class's section."""
        number = self.set_numbers().get((name, field), {}).get(value)
        if number is None:
            return "<unresolved parent set>"
        return f"S{number}"

    def set_sort(self, name: str, field: str, value: str) -> tuple:
        members = self._set_members(name, field, value)
        if members is None:
            return (9,)
        return (4,) + tuple(sort for sort, _ in members)

    # --- labels (prompts/datastore-integrity prompt 09d) ---------------------------------------

    def labelled(self) -> frozenset:
        """The sharded classes that another class's key refers to: the derived rows. A reference
        to one is printed as a label, ``<Class>#<n>``; every other reference is printed from the
        parent's key."""
        if self._labelled is None:
            targets = set()
            for cls in self.inventory.classes.values():
                targets.update(cls.parents.values())
                for members in cls.parent_sets.values():
                    targets.update(members.values())
            self._labelled = frozenset(
                n
                for n in targets
                if n in self.inventory.classes and not self.inventory[n].replicated
            )
        return self._labelled

    def shown(self, name: str, members: List[Record]) -> List[Record]:
        """The records of one tag set that the report lists in full."""
        return members if (self.verbose or name == "store_tag") else members[:MAX_SHOWN]

    def groups(self, name: str) -> List[Tuple[Tuple[str, ...], List[Record]]]:
        """The tag sets of ``name`` in order, each with its records in physical order."""
        if name not in self._groups:
            fields, _ = self.line_fields(name)
            grouped: Dict[Tuple[str, ...], List[Record]] = {}
            for record in self.cls(name).records:
                grouped.setdefault(record.tags, []).append(record)
            self._groups[name] = [
                (tags, sorted(members, key=lambda r: self.record_sort(name, r, fields)))
                for tags, members in sorted(grouped.items())
            ]
        return self._groups[name]

    def ordinals(
        self, name: str
    ) -> Tuple[Dict[int, int], Dict[str, int], List[Record]]:
        """A record's ordinal in its class's verbose listing: tag set by tag set, in physical
        order, from 1. Returns (id of record -> ordinal, reference digest -> the lowest ordinal of
        a record it names, the records by ordinal - 1). It does not depend on a serial.
        """
        if name not in self._ordinals:
            of_record: Dict[int, int] = {}
            of_digest: Dict[str, int] = {}
            listing: List[Record] = []
            for _, members in self.groups(name):
                for record in members:
                    listing.append(record)
                    of_record[id(record)] = len(listing)
                    of_digest.setdefault(
                        self.inventory.digest_of(name, record), len(listing)
                    )
            self._ordinals[name] = (of_record, of_digest, listing)
        return self._ordinals[name]

    def set_numbers(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """(class, field) -> set digest -> the set's ordinal ``n`` of ``S<n>``: in physical order,
        from 1, through every class that holds sets, so that a label is unique in the report.
        """
        if self._set_numbers is None:
            out: Dict[Tuple[str, str], Dict[str, int]] = {}
            count = 0
            for name in sorted(self.inventory.classes):
                for field in self.cls(name).parent_sets:
                    digests = {
                        r.key[field]
                        for r in self.cls(name).records
                        if self._set_members(name, field, r.key[field]) is not None
                    }
                    ordered = sorted(
                        digests, key=lambda d: (self.set_sort(name, field, d), d)
                    )
                    out[(name, field)] = {
                        d: count + i for i, d in enumerate(ordered, start=1)
                    }
                    self._set_digests[(name, field)] = {
                        count + i: d for i, d in enumerate(ordered, start=1)
                    }
                    count += len(ordered)
            self._set_numbers = out
        return self._set_numbers

    def needed(self) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, str], List[int]]]:
        """What the report must define beyond the records it lists: (class -> ordinals of the rows
        that a listed record refers to and the report has not listed, and so on for those rows,
        (class, field) -> the ordinals ``n`` of the sets it prints). Closed under reference. With
        ``verbose`` every record is listed, so only the sets remain."""
        if self._needed is None:
            numbers = self.set_numbers()
            held = {
                (name, field)
                for name in self.inventory.classes
                for field in self.cls(name).parent_sets
            }
            sets: Dict[Tuple[str, str], set] = {key: set() for key in held}
            extra: Dict[str, set] = {}
            visited = set()
            work = []
            for name in self.inventory.classes:
                for _, members in self.groups(name):
                    for record in self.shown(name, members):
                        key = (name, self.ordinals(name)[0][id(record)])
                        visited.add(key)
                        work.append(key)
            listed = set(visited)
            done = set()
            if self.verbose:
                # every record is listed, so only the sets remain, and all of them are printed
                work = []
                for key, digests in self.set_numbers().items():
                    sets[key].update(digests.values())
            while work:
                name, number = work.pop()
                cls = self.cls(name)
                record = self.ordinals(name)[2][number - 1]
                for field, value in record.key.items():
                    targets = []
                    if field in cls.parent_sets:
                        digest = value
                        if digest in numbers.get((name, field), {}) and (
                            (name, field, digest) not in done
                        ):
                            done.add((name, field, digest))
                            sets[(name, field)].add(numbers[(name, field)][digest])
                            classes = cls.parent_sets[field]
                            for member in cls.members.get(field, {}).get(digest, ()):
                                targets.extend(zip(classes.values(), member))
                    elif field in cls.parents:
                        targets.append((self.parent_class(name, record, field), value))
                    for of, reference in targets:
                        if of not in self.labelled() or reference is None:
                            continue
                        found = self.ordinals(of)[1].get(reference)
                        if found is None or (of, found) in visited:
                            continue
                        visited.add((of, found))
                        if (of, found) not in listed:
                            extra.setdefault(of, set()).add(found)
                        work.append((of, found))
            self._needed = (
                {n: sorted(v) for n, v in extra.items()},
                {key: sorted(v) for key, v in sets.items()},
            )
        return self._needed

    def set_line(self, name: str, field: str, number: int) -> str:
        self.set_numbers()
        digest = self._set_digests[(name, field)][number]
        return f"parent set S{number}: {self.members_text(name, field, digest)}"

    # --- one field of one record -------------------------------------------------------------

    def value_text(self, name: str, record: Record, field: str) -> str:
        value = record.key[field]
        if field in self.cls(name).parent_sets:
            return self.set_text(name, field, value)
        if field in self.cls(name).parents:
            return self.ref_text(self.parent_class(name, record, field), value)
        return self.leaf_text(name, field, value)

    def value_sort(self, name: str, record: Record, field: str) -> tuple:
        value = record.key[field]
        if field in self.cls(name).parent_sets:
            return self.set_sort(name, field, value)
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
        if name in self.labelled():
            # beyond the first double space, where a parser of the fields stops
            text += f"  #{self.ordinals(name)[0][id(record)]}"
        if record.value_count is not None:
            text += f"  | values: {record.value_count:,}"
        return text

    def record_sort(self, name: str, record: Record, fields: Sequence[str]) -> tuple:
        return tuple(self.value_sort(name, record, f) for f in fields) + (
            record.canonical_json(),
        )


def _class_lines(renderer: _Renderer, name: str, carried: frozenset) -> List[str]:
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

        ordered = renderer.groups(name)
        for number, (tags, members) in enumerate(ordered, start=1):
            if cls.tagged:
                labels = ", ".join(tags) if len(tags) > 0 else "(no tags)"
                lines.append(
                    f"{_DETAIL_INDENT}tag set {number} of {len(ordered)} "
                    f"({_plural(len(members), 'record')}): {labels}"
                )
            shown = renderer.shown(name, members)
            for record in shown:
                line = renderer.record_line(name, record, fields)
                if name == "store_tag" and record.key.get("label") not in carried:
                    line += "  (carried by no record)"
                lines.append(f"{_RECORD_INDENT}- {line}")
            if len(members) > len(shown):
                lines.append(
                    f"{_RECORD_INDENT}... and {len(members) - len(shown):,} more"
                )

        # a row that a listed record refers to, and that the report has not listed, is defined
        # here, once, under a prefix that is not a record line's; and so is every parent set.
        # The record that names it is often in a later section, so the header does not say where
        extra, sets = renderer.needed()
        if len(extra.get(name, ())) > 0:
            _, _, listing = renderer.ordinals(name)
            lines.append(
                f"{_DETAIL_INDENT}{_plural(len(extra[name]), 'row')} not shown above, and "
                f"referred to elsewhere in this report:"
            )
            for number in extra[name]:
                record = listing[number - 1]
                labels = ", ".join(record.tags) if len(record.tags) > 0 else "(no tags)"
                tagged = f"  tags: {labels}" if cls.tagged else ""
                lines.append(
                    f"{_RECORD_INDENT}> {renderer.record_line(name, record, fields)}{tagged}"
                )
        for field in cls.parent_sets:
            for number in sets.get((name, field), ()):
                lines.append(
                    f"{_DETAIL_INDENT}{renderer.set_line(name, field, number)}"
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
            lines.extend(_class_lines(renderer, name, carried))

    return "\n".join(lines)
