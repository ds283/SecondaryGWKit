"""
Check the proposed fix shapes for [00-tagged-read-batch-joins-an-unselected-alias] and the
QuadSourceIntegral.read_batch exit-serial read-back, without editing any tracked file.

The factory module's source is read, the two proposed one-line edits are applied **in memory**,
and the patched module is executed into a throwaway namespace. probe_01's store and cases are then
run against the patched read_batch.

  edit 1 (tag join):   tab.c.parent_serial == query.c.serial  ->  tab.c.parent_serial == table.c.serial
  edit 2 (exit ids):   select q_exit_tab.c.serial / r_exit_tab.c.serial as q_exit_serial /
                       r_exit_serial, and give the rebuilt wavenumber_exit_time those, not the
                       wavenumber serials
"""

import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import probe_01_tagged_read_batch as p1  # noqa: E402
from common import Ref, StandInModelProxy, Store  # noqa: E402
from MetadataConcepts import store_tag, tolerance  # noqa: E402

SRC = ROOT / "Datastore" / "SQL" / "ObjectFactories" / "QuadSourceIntegral.py"


def patched_module():
    text = SRC.read_text()
    edits = [
        (
            "tab.c.parent_serial == query.c.serial",
            "tab.c.parent_serial == table.c.serial",
        ),
        (
            'q_exit_tab.c.stepping.label("q_stepping"),',
            'q_exit_tab.c.serial.label("q_exit_serial"),\n'
            '                q_exit_tab.c.stepping.label("q_stepping"),',
        ),
        (
            'r_exit_tab.c.stepping.label("r_stepping"),',
            'r_exit_tab.c.serial.label("r_exit_serial"),\n'
            '                r_exit_tab.c.stepping.label("r_stepping"),',
        ),
        (
            '"store_id": row.q_serial,\n                "z_exit": row.q_z_exit,',
            '"store_id": row.q_exit_serial,\n                "z_exit": row.q_z_exit,',
        ),
        (
            '"store_id": row.r_serial,\n                "z_exit": row.r_z_exit,',
            '"store_id": row.r_exit_serial,\n                "z_exit": row.r_z_exit,',
        ),
    ]
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    mod = types.ModuleType("patched_QuadSourceIntegral")
    exec(compile(text, str(SRC) + " [patched in memory]", "exec"), mod.__dict__)
    return mod


def main():
    mod = patched_module()
    s = Store()
    p1.populate(s)
    tags = {
        "A": store_tag(store_id=1, label="A"),
        "B": store_tag(store_id=2, label="B"),
    }
    expected = {(): {1, 2, 3}, ("A",): {1, 2}, ("A", "B"): {1}}
    for names, want in expected.items():
        payload = dict(
            model=StandInModelProxy(),
            policy=Ref(1),
            k=p1.exit_time(11, 1, 1.0e5),
            q=p1.exit_time(12, 2, 2.0e5),
            r=p1.exit_time(13, 3, 3.0e5),
            z_response=None,
            z_source_max=None,
            atol=tolerance(store_id=1, log10_tol=-10.0),
            rtol=tolerance(store_id=2, log10_tol=-9.0),
            tags=[tags[n] for n in names],
        )
        objs = mod.sqla_QuadSourceIntegral_factory.read_batch(
            payload, s.conn, s.tables["QuadSourceIntegral"], s.tables
        )
        got = {o.store_id for o in objs}
        exits = sorted({(o._q_exit.store_id, o._r_exit.store_id) for o in objs})
        print(
            f"patched read_batch tags={list(names)}: serials={sorted(got)} correct={got == want}; "
            f"(q_exit, r_exit) store_ids={exits} (stored: (12, 13))"
        )


if __name__ == "__main__":
    main()
