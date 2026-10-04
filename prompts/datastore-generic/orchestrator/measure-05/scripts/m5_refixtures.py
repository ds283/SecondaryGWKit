"""Check that orphan tag rows can stand in for absent tables in the two re-fixtured tests."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import sqlalchemy as sqla

from Datastore.SQL.Datastore import _factories
from Datastore.SQL.schema import build_schema
from Datastore.store_inventory import read_inventory
from Datastore.tests.real_store_fixtures import build_full_store
from extract_common import available_run_labels
from tools.inventory_report import MAX_SHOWN

SCRATCH = Path(__file__).resolve().parents[1] / "tmp"
SCRATCH.mkdir(exist_ok=True)
built = build_schema(sqla.MetaData(), _factories)


def orphan_tag(table):
    t = built.tables[table]
    parent = [
        fk.parent.name for fk in t.foreign_keys if fk.column.table.name != "store_tag"
    ][0]
    tag = [
        fk.parent.name for fk in t.foreign_keys if fk.column.table.name == "store_tag"
    ][0]
    return f'INSERT INTO "{table}" ({parent}, {tag}) VALUES (999, 1)'


print(
    "tag table columns:",
    {
        t: [c.name for c in built.tables[t].columns]
        for t in ["QuadSourceIntegral_tags", "TkNumeric_tags"]
    },
)

with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
    s = build_full_store(
        Path(d), extra_sql={1: [orphan_tag("QuadSourceIntegral_tags")]}
    )
    inv = read_inventory(s.primary)
    print("(a) problems:", inv.problems)
    print("(a) labels:", available_run_labels(SimpleNamespace(primary=s.primary)))

absent = [
    "TkNumeric_tags",
    "TkWKB_tags",
    "GkNumeric_tags",
    "GkWKB_tags",
    "GkSource_tags",
    "QuadSource_tags",
    "QuadSourceIntegral_tags",
]
with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
    extra = {sh: [orphan_tag("QuadSourceIntegral_tags")] for sh in range(1, 7)}
    extra[1] = [orphan_tag(t) for t in absent]
    s = build_full_store(Path(d), shards=7, extra_sql=extra)
    inv = read_inventory(s.primary)
    print(
        "(b) MAX_SHOWN",
        MAX_SHOWN,
        "total problems",
        len(inv.problems),
        "QuadSourceIntegral problems",
        len(inv["QuadSourceIntegral"].problems),
    )
