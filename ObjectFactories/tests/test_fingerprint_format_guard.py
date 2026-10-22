"""Guard the fingerprint format against spec changes (runregistry-extraction prompt 07; README
§6.1, Q5; docs/runregistry-extraction-audit.md §3.5).

`config.fingerprint.FINGERPRINT_FORMAT` is a number the author bumps when what the inventory
records changes. Nothing makes the bump happen, and the golden fingerprint says only that the data
changed, not that the format was forgotten. This module pins, for each format that has a pin
(`config.fingerprint.SPEC_DIGESTS`), the SHA-256 of the canonical JSON of every
`inventory_spec()` in the registry, in registry order, as `[class, digest]` pairs. A change to any
spec then fails `TestFormatGuard` naming the class, and says to bump the format.

The functions are the recipe of RunRegistryKit's `docs/adoption/README.md` §6 (nothing of the guard
is shipped by the package; it is this repository's test). `plain` writes every mapping key as its
own JSON, so that the polymorphic `Parent.types` keys of `BackgroundModel` and
`wavenumber_exit_time`, which are `int`, stay apart from the string `"0"`: `canonical_json` refuses
a key that is not a string, and a bare `str(k)` would make the two alike and miss a change of a
key's type. `test_a_key_keeps_its_type` pins that conversion.

Since DatastoreKit `v0.5.0` the recipe carries one amendment, given in that file's section "The Q5
guard under DatastoreKit `v0.5.0`" (RunRegistryKit `70ee1e8`): `plain` leaves out an empty
`extra_values`. With it every pin in `SPEC_DIGESTS[6]` holds under `v0.5.0`, so the new field is not
a format change, while a spec that comes to declare an extra table moves its digest and is one
(dk-v050-adoption prompt 01; `test_an_empty_extra_values_is_left_out`).

Nothing is opened: no store, nothing under `var/`, no Ray.
"""

import dataclasses
import hashlib
import json
import unittest
from collections.abc import Mapping

from datastorekit.store_inventory import InventorySpec, canonical_json, inventory_specs

from config.datastore import factories
from config.fingerprint import FINGERPRINT_FORMAT, SPEC_DIGESTS


def plain(value):
    """A spec as JSON: dataclasses as objects, sequences as lists, and every mapping key as the
    JSON of the key, so that 1 (`1`) and "1" (`"1"`) stay apart. An empty `extra_values` is left
    out, so that a spec declaring no extra table is what it was before DatastoreKit `v0.5.0` added
    the field."""
    if dataclasses.is_dataclass(value):
        return {
            f.name: plain(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if not (f.name == "extra_values" and len(getattr(value, f.name)) == 0)
        }
    if isinstance(value, Mapping):
        return {json.dumps(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def spec_digests(factories) -> list:
    """[class, SHA-256 of its spec's canonical JSON] for each spec, in registry order."""
    return [
        [name, hashlib.sha256(canonical_json(plain(spec)).encode()).hexdigest()]
        for name, spec in inventory_specs(factories).items()
    ]


class TestFormatGuard(unittest.TestCase):
    def test_every_spec_is_pinned_for_the_current_format(self):
        if FINGERPRINT_FORMAT not in SPEC_DIGESTS:
            self.fail(
                f"FINGERPRINT_FORMAT is {FINGERPRINT_FORMAT} but config.fingerprint.SPEC_DIGESTS "
                f"has no pin for it (it has {sorted(SPEC_DIGESTS)}): the format was bumped "
                "without a pin. Generate one with spec_digests(factories) from this module and "
                "add it beside the old ones, never over them, with a new golden file."
            )
        pinned = SPEC_DIGESTS[FINGERPRINT_FORMAT]
        current = spec_digests(factories)
        if current == pinned:
            return

        pinned_by_name = dict(pinned)
        current_by_name = dict(current)
        changed = [
            n
            for n in current_by_name
            if n in pinned_by_name and current_by_name[n] != pinned_by_name[n]
        ]
        added = [n for n in current_by_name if n not in pinned_by_name]
        removed = [n for n in pinned_by_name if n not in current_by_name]
        lines = [
            f"the inventory specs no longer match the pin for format {FINGERPRINT_FORMAT}:"
        ]
        lines += [f"  spec changed: {n}" for n in changed]
        lines += [f"  class added to the registry with a spec: {n}" for n in added]
        lines += [
            f"  class removed from the registry or its spec: {n}" for n in removed
        ]
        if not (changed or added or removed):
            lines.append(
                "  the classes and digests agree, but the registry order differs"
            )
        lines.append(
            "A spec changed, so bump FINGERPRINT_FORMAT in config/fingerprint.py and add a new "
            "pin to SPEC_DIGESTS and a new golden file beside the old ones, never over them."
        )
        self.fail("\n".join(lines))

    def test_a_key_keeps_its_type(self):
        self.assertNotEqual(plain({0: "a"}), plain({"0": "a"}))
        self.assertNotEqual(
            canonical_json(plain({0: "a"})), canonical_json(plain({"0": "a"}))
        )

    def test_an_empty_extra_values_is_left_out(self):
        empty = plain(InventorySpec())
        declared = plain(InventorySpec(extra_values={"t": "c"}))
        self.assertNotIn("extra_values", empty)
        self.assertIn("extra_values", declared)
        self.assertNotEqual(canonical_json(empty), canonical_json(declared))


if __name__ == "__main__":
    unittest.main()
