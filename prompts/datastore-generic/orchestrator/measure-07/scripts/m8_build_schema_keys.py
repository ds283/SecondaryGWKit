"""M8: build_schema's new keys: refusals of bad declarations, and the real registry's records.

usage (from the worktree, PYTHONPATH=.): python m8_build_schema_keys.py
"""
import sqlalchemy as sqla

from config.datastore import factories
from Datastore.SQL.schema import build_schema


def factory(**declared):
    class F:
        @staticmethod
        def register():
            return {
                "version": False,
                "timestamp": False,
                "columns": [
                    sqla.Column("plain", sqla.Integer),
                    sqla.Column("owner_serial", sqla.Integer, sqla.ForeignKey("Owner.serial")),
                    sqla.Column(
                        "two_keys",
                        sqla.Integer,
                        sqla.ForeignKey("Owner.serial"),
                        sqla.ForeignKey("Other.serial"),
                    ),
                    sqla.Column("flag", sqla.Boolean),
                ],
                **declared,
            }

    return F


class Owner:
    @staticmethod
    def register():
        return {"version": False, "timestamp": False, "columns": []}


cases = {
    "owner_column absent": {"owner_column": "nonesuch"},
    "owner_column with no foreign key": {"owner_column": "plain"},
    "owner_column with two foreign keys": {"owner_column": "two_keys"},
    "monotone_flags naming an absent column": {"monotone_flags": ("flag", "nonesuch")},
    "validated_column absent": {"validated_column": "nonesuch"},
    "all three well declared": {
        "owner_column": "owner_serial",
        "monotone_flags": ("flag",),
        "validated_column": "flag",
    },
}
for name, declared in cases.items():
    try:
        built = build_schema(
            sqla.MetaData(), {"Owner": Owner, "Other": Owner, "Member": factory(**declared)}
        )
    except ValueError as e:
        print(f"{name}: ValueError: {e}")
    else:
        r = built.records["Member"]
        print(
            f"{name}: accepted: owner_column={r['owner_column']!r}, "
            f"monotone_flags={r['monotone_flags']!r}, validated_column={r['validated_column']!r}"
        )

print()
print("the real registry: every record with a table, the three keys")
built = build_schema(sqla.MetaData(), factories)
declared = {}
for cls_name, r in built.records.items():
    if r["table"] is None:
        print(f"  {cls_name}: no table; keys present: {[k for k in ('owner_column', 'monotone_flags', 'validated_column') if k in r]}")
        continue
    values = (r["owner_column"], r["monotone_flags"], r["validated_column"])
    if values != (None, (), None):
        declared[cls_name] = values
print("  non-default:", declared)
print(
    "  every other record:",
    sorted({(r["owner_column"], r["monotone_flags"], r["validated_column"]) for n, r in built.records.items() if n not in declared}),
)
expected = {
    "redshift": (None, ("source", "response"), None),
    "wavenumber": (None, ("source", "response"), None),
    "BackgroundModel": (None, (), "validated"),
    "BackgroundModelValue": ("model_serial", (), None),
}
print("  exactly the declared values:", declared == expected)
