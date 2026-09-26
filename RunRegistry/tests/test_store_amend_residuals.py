"""The two residuals prompt 06 fixes in `amend_sidecar` (store-retirement prompt 06):

- **R14.** The identical-value refusal (`amend_sidecar`'s ninth check) now compares the current
  and the given value by their canonical JSON text (`json.dumps(…, sort_keys=True)`), not Python
  `==`. `true`, `1` and `1.0` are three different values, and each amendment between them is
  accepted, keeping its own type; an object with its keys in another order is still refused,
  because its canonical text is unchanged.
- **R15.** A value shaped exactly like the `before`/`after` marker itself — `{"present": ...}` —
  is taken through add, replace and remove, and read back unambiguously, so that a later change
  which flattened the wrapper (returning the bare value instead of `{"present": True, "value":
  …}`) would be caught here.

This module imports `AmendTestCase` and `REASON` from `test_store_amend` rather than copying
them, so it shares the same lightweight placeholder stores, the same `runs_root` discipline
(every `amend_sidecar` call passes it, every command passes `--runs-root`), and nothing under
`var/` is opened. `test_store_amend.py` itself is not touched by this module. Seeding calls below
use `AmendTestCase.amend`'s default reason, `REASON`; only the amendment under test is given a
reason that says what it does.
"""

from RunRegistry import stores
from RunRegistry.tests.test_store_amend import AmendTestCase, REASON


# =================================================================================================
# R14 — the identical-value refusal compares canonical JSON text, not Python `==`


class TestTypeStrictComparison(AmendTestCase):
    """`assertEqual` alone cannot tell `1` from `True`, or `1` from `1.0`: each pair is equal
    under Python `==`, at any depth, but their canonical JSON texts differ, so each amendment
    below is accepted, and the field keeps the type it was given, not the type it had before.
    """

    def test_1_to_true(self):
        primary = self.a_registry_store()
        self.amend(primary, "n", REASON, value=1)
        before = self.state()
        result = self.amend(primary, "n", "change 1 to true", value=True)
        self.assertEqual(result["before"], 1)
        self.assertEqual(result["after"], True)
        entry = self.sidecar(primary)["history"][-1]
        self.assertIs(type(entry["before"]["value"]), int)
        self.assertIs(type(entry["after"]["value"]), bool)
        self.assertOnlySidecarChanged(primary, before)
        # the raw bytes on disk, not just the value read back through json.loads
        text = stores.sidecar_path(primary).read_text()
        self.assertRegex(text, r'"n":\s*true')

    def test_1_to_1_0(self):
        primary = self.a_registry_store()
        self.amend(primary, "n", REASON, value=1)
        before = self.state()
        result = self.amend(primary, "n", "change 1 to 1.0", value=1.0)
        self.assertEqual(result["before"], 1)
        self.assertEqual(result["after"], 1.0)
        entry = self.sidecar(primary)["history"][-1]
        self.assertIs(type(entry["before"]["value"]), int)
        self.assertIs(type(entry["after"]["value"]), float)
        self.assertOnlySidecarChanged(primary, before)
        text = stores.sidecar_path(primary).read_text()
        self.assertRegex(text, r'"n":\s*1\.0')

    def test_true_to_1(self):
        primary = self.a_registry_store()
        self.amend(primary, "n", REASON, value=True)
        before = self.state()
        result = self.amend(primary, "n", "change true to 1", value=1)
        self.assertEqual(result["before"], True)
        self.assertEqual(result["after"], 1)
        entry = self.sidecar(primary)["history"][-1]
        self.assertIs(type(entry["before"]["value"]), bool)
        self.assertIs(type(entry["after"]["value"]), int)
        self.assertOnlySidecarChanged(primary, before)
        # exactly "1", never "true" and never "1.0", where the field itself is printed
        text = stores.sidecar_path(primary).read_text()
        self.assertRegex(text, r'"n":\s*1[,\n]')
        self.assertNotRegex(text, r'"n":\s*(true|1\.0)')

    def test_dict_value_1_to_true(self):
        primary = self.a_registry_store()
        self.amend(primary, "n", REASON, value={"k": 1})
        before = self.state()
        result = self.amend(primary, "n", "change k from 1 to true", value={"k": True})
        self.assertEqual(result["before"], {"k": 1})
        self.assertEqual(result["after"], {"k": True})
        entry = self.sidecar(primary)["history"][-1]
        self.assertIs(type(entry["before"]["value"]["k"]), int)
        self.assertIs(type(entry["after"]["value"]["k"]), bool)
        self.assertOnlySidecarChanged(primary, before)
        text = stores.sidecar_path(primary).read_text()
        self.assertRegex(text, r'"k":\s*true')

    def test_list_value_1_to_true(self):
        primary = self.a_registry_store()
        self.amend(primary, "n", REASON, value=[1])
        before = self.state()
        result = self.amend(
            primary, "n", "change the one entry from 1 to true", value=[True]
        )
        self.assertEqual(result["before"], [1])
        self.assertEqual(result["after"], [True])
        entry = self.sidecar(primary)["history"][-1]
        self.assertIs(type(entry["before"]["value"][0]), int)
        self.assertIs(type(entry["after"]["value"][0]), bool)
        self.assertOnlySidecarChanged(primary, before)
        text = stores.sidecar_path(primary).read_text()
        self.assertRegex(text, r'"n":\s*\[\s*true')


class TestStillRefusedWithTheTreeUnchanged(AmendTestCase):
    """A value whose canonical JSON text does not change is still refused, whatever its Python
    identity: an unchanged scalar, and an object with its keys in another order (whose text,
    under `sort_keys=True`, is exactly the same either way)."""

    def test_1_to_1(self):
        primary = self.a_registry_store()
        self.amend(primary, "n", REASON, value=1)
        self.refused_amend(
            primary, "identical, after a JSON round trip", field="n", value=1
        )

    def test_an_object_with_its_keys_reordered(self):
        primary = self.a_registry_store()
        self.amend(primary, "n", REASON, value={"a": 1, "b": 2})
        self.refused_amend(
            primary,
            "identical, after a JSON round trip",
            field="n",
            value={"b": 2, "a": 1},
        )


class TestCommandLineTypeChange(AmendTestCase):
    """`store amend … --field n --json true` on a field holding `1` succeeds, and the same
    command run again — now genuinely identical — is refused."""

    def test_true_over_1_then_refused_the_second_time(self):
        primary = self.a_registry_store()
        self.amend(primary, "n", REASON, value=1)
        runs = ["--runs-root", self.root]

        report, out, _ = self.command(
            "store",
            "amend",
            primary,
            "--field",
            "n",
            "--json",
            "true",
            "--reason",
            "change n from 1 to true",
            *runs,
        )
        self.assertEqual(report["code"], 0)
        self.assertIn(">> amended:", out)
        self.assertEqual(self.sidecar(primary)["n"], True)

        err = self.refused(
            "store",
            "amend",
            primary,
            "--field",
            "n",
            "--json",
            "true",
            "--reason",
            "run it again",
            *runs,
        )
        self.assertIn("identical, after a JSON round trip", err)


# =================================================================================================
# R15 — a value shaped like the marker itself, through add, replace, remove and read-back


class TestMarkerShapedValue(AmendTestCase):
    """One field, `m`, whose value at every step is itself shaped exactly like an amend marker —
    `{"present": False}` or `{"present": True, "value": …}` — so that a wrapper which lost its
    outer layer (returning the bare value where `_amend_slot` should return `{"present": True,
    "value": <the bare value>}`) is caught by an exact-equality check of the whole marker, not
    merely of the value it wraps.
    """

    def test_add_replace_remove_and_read_back(self):
        primary = self.a_registry_store()

        # 1. add: m was absent, so `before` says so, and `after` wraps the marker-shaped value
        with self.subTest("add"):
            before_state = self.state()
            result = self.amend(primary, "m", "add m", value={"present": False})
            self.assertIsNone(result["before"])
            self.assertEqual(result["after"], {"present": False})
            entry = self.sidecar(primary)["history"][-1]
            self.assertEqual(entry["before"], {"present": False})
            self.assertEqual(
                entry["after"], {"present": True, "value": {"present": False}}
            )
            self.assertOnlySidecarChanged(primary, before_state)

        # 2. replace: `before` must be exactly the previous `after`, one level unwrapped
        with self.subTest("replace"):
            before_state = self.state()
            result = self.amend(
                primary, "m", "replace m", value={"present": True, "value": 1}
            )
            self.assertEqual(result["before"], {"present": False})
            self.assertEqual(result["after"], {"present": True, "value": 1})
            entry = self.sidecar(primary)["history"][-1]
            self.assertEqual(
                entry["before"], {"present": True, "value": {"present": False}}
            )
            self.assertEqual(
                entry["after"],
                {"present": True, "value": {"present": True, "value": 1}},
            )
            self.assertOnlySidecarChanged(primary, before_state)

        # 3. remove: `before` is exactly the previous `after`, again one level unwrapped
        with self.subTest("remove"):
            before_state = self.state()
            result = self.amend(primary, "m", "remove m", remove=True)
            self.assertEqual(result["before"], {"present": True, "value": 1})
            self.assertIsNone(result["after"])
            entry = self.sidecar(primary)["history"][-1]
            self.assertEqual(
                entry["before"],
                {"present": True, "value": {"present": True, "value": 1}},
            )
            self.assertEqual(entry["after"], {"present": False})
            self.assertOnlySidecarChanged(primary, before_state)

        # 4. read back: the sidecar is problem-free, and a small walk over the three `amend`
        # entries, unwrapping each marker one level, reconstructs the sequence unambiguously
        with self.subTest("read back"):
            reading = stores.read_sidecar(primary)
            self.assertTrue(reading.ok, reading.problems)
            self.assertEqual(stores._history_problems(reading.fields["history"]), [])

            amends = [e for e in reading.fields["history"] if e["operation"] == "amend"]
            self.assertEqual(len(amends), 3)

            # each entry's before is exactly the previous entry's after: a wrapper that lost its
            # outer layer on one side only would still show up here, even where a single step's
            # own assertions, above, happened not to catch it
            for index in range(1, len(amends)):
                self.assertEqual(
                    amends[index]["before"],
                    amends[index - 1]["after"],
                    f"amend entry #{index}'s before",
                )

            def unwrap(marker):
                return marker["value"] if marker["present"] else None

            markers = [amends[0]["before"]] + [entry["after"] for entry in amends]
            self.assertEqual(
                [unwrap(marker) for marker in markers],
                [None, {"present": False}, {"present": True, "value": 1}, None],
            )

            report, out, _ = self.command(
                "store", "show", primary, "--runs-root", self.root
            )
            self.assertEqual(report["code"], 0)
            self.assertIn("problems: none", out)
