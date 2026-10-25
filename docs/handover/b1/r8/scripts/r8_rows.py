"""handover prompt 12, section 6.5: R8's rows, read-only.

    ./venv/bin/python -I var/handover-12/scripts/r8_rows.py \
        var/datastores/handover-12-reference.sqlite \
        var/handover-06/extract/extract_tensor_source_data/LambdaCDM/csv

Every file of the store is opened by sqlite3 with a `file:...?mode=ro` URI; nothing is written.
The shards are read from the primary's own `shards` table.

1. Every QuadSourceIntegral row: has metadata["partition"]["handover"] with Tq and Tr, and no key
   containing "clamp" anywhere in its metadata (clamp_gaps_log1pz, max_clamp_gap_log1pz, ...).
2. The handover_continuity records (dT_rel, dTprime_rel) by k and node, de-duplicated across rows
   (each k's records must be identical in every row that carries them), with a summary by node.
3. Every QuadSource row's z extent and node count, against the R3-equal record: the R4 store's
   extract_tensor_source_data CSVs (handover 06; R4 reproduced R3's fingerprint 39809dca). The
   nodes added and removed, and whether every column is bit-identical at the nodes both hold.
"""

import csv
import json
import math
import sqlite3
import sys
from pathlib import Path


def ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def find_keys(obj, needle, path=""):
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else key
            if needle in key.lower():
                hits.append(here)
            hits.extend(find_keys(value, needle, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits.extend(find_keys(value, needle, f"{path}[{i}]"))
    return hits


VALUE_COLUMNS = [
    "source",
    "undiff",
    "diff",
    "analytic_source_rad",
    "analytic_undiff_rad",
    "analytic_diff_rad",
    "analytic_source_w",
    "analytic_undiff_w",
    "analytic_diff_w",
]


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def main(primary: Path, r3_csv_dir: Path) -> int:
    with ro(primary) as conn:
        shards = [
            primary.parent / Path(name).name
            for (name,) in conn.execute("SELECT filename FROM shards ORDER BY serial")
        ]

    # replicated tables, from shard 0
    with ro(shards[0]) as conn:
        z_of = dict(conn.execute("SELECT serial, z FROM redshift"))
        source_grid = sorted(
            (z for (z,) in conn.execute("SELECT z FROM redshift WHERE source = 1")),
            reverse=True,
        )
        k_of_exit = dict(
            conn.execute(
                "SELECT e.serial, w.k_inv_Mpc FROM wavenumber_exit_time e "
                "JOIN wavenumber w ON w.serial = e.wavenumber_serial"
            )
        )

    problems = []

    # ---- 1. and 2.: QuadSourceIntegral rows
    n_rows = 0
    n_handover = 0
    clamp_hits = []
    continuity = {}  # k -> (z_init, z_switch, records)
    for s, path in enumerate(shards):
        with ro(path) as conn:
            for serial, q_exit, r_exit, meta in conn.execute(
                "SELECT serial, q_wavenumber_exit_serial, r_wavenumber_exit_serial, metadata "
                "FROM QuadSourceIntegral"
            ):
                n_rows += 1
                data = json.loads(meta)
                hits = find_keys(data, "clamp")
                if hits:
                    clamp_hits.append((s, serial, hits))
                handover = data.get("partition", {}).get("handover")
                if (
                    isinstance(handover, dict)
                    and set(handover) >= {"Tq", "Tr"}
                    and all("handover_continuity" in handover[t] for t in ("Tq", "Tr"))
                ):
                    n_handover += 1
                else:
                    problems.append(f"shard {s} QSI serial {serial}: no complete partition.handover")
                    continue
                for label, exit_serial in (("Tq", q_exit), ("Tr", r_exit)):
                    k = k_of_exit[exit_serial]
                    rec = handover[label]
                    entry = (rec["z_init"], rec["z_switch"], rec["handover_continuity"])
                    if k in continuity:
                        if continuity[k] != entry:
                            problems.append(
                                f"shard {s} QSI serial {serial}: {label} k={k:.5g} continuity "
                                f"record differs from another row's"
                            )
                    else:
                        continuity[k] = entry

    print("== 1. QuadSourceIntegral rows")
    print(f"rows: {n_rows}; with partition.handover (Tq and Tr, each with handover_continuity): {n_handover}")
    print(f"rows with any metadata key containing 'clamp': {len(clamp_hits)}")
    for s, serial, hits in clamp_hits[:20]:
        print(f"  shard {s} serial {serial}: {hits}")

    print()
    print("== 2. handover_continuity, by k and node (identical in every row that carries the k)")
    by_node = {}
    for k in sorted(continuity):
        z_init, z_switch, records = continuity[k]
        between = [z for z in source_grid if z_switch < z < z_init]
        first_below = next((z for z in source_grid if z < z_init), None)
        print(
            f"k = {k:.5g}/Mpc: z_init = {z_init!r}, z_switch = {z_switch!r}; "
            f"z_switch is the first source-grid node below z_init: {first_below == z_switch}; "
            f"source-grid nodes strictly between: {len(between)}; nodes recorded: {len(records)}"
        )
        for i, r in enumerate(records):
            print(
                f"  node {i}: z = {r['z']!r:>22}  dT_rel = {r['dT_rel']: .6e}  "
                f"dTprime_rel = {r['dTprime_rel']: .6e}"
            )
            by_node.setdefault(i, []).append(r)
    print("summary by node, over k:")
    for i in sorted(by_node):
        dT = [abs(r["dT_rel"]) for r in by_node[i]]
        dTp = [abs(r["dTprime_rel"]) for r in by_node[i]]
        print(
            f"  node {i} ({len(dT)} k): |dT_rel| min {min(dT):.3e} max {max(dT):.3e}; "
            f"|dTprime_rel| min {min(dTp):.3e} max {max(dTp):.3e}"
        )

    # ---- 3.: QuadSource rows against R4's (R3-equal) extract
    print()
    print("== 3. QuadSource rows: R8 against R4's extract (R3-equal fingerprint 39809dca)")
    r3_files = {p.name.split("-q-serial")[0]: p for p in r3_csv_dir.glob("*.csv")}
    matched = set()
    total_added = 0
    for s, path in enumerate(shards):
        with ro(path) as conn:
            rows = conn.execute(
                "SELECT serial, q_wavenumber_exit_serial, r_wavenumber_exit_serial, z_samples, validated "
                "FROM QuadSource ORDER BY serial"
            ).fetchall()
            for serial, q_exit, r_exit, z_samples, validated in rows:
                q, r = k_of_exit[q_exit], k_of_exit[r_exit]
                values = {}
                for row in conn.execute(
                    f"SELECT z_serial, {', '.join(VALUE_COLUMNS)} FROM QuadSourceValue "
                    "WHERE parent_serial = ?",
                    (serial,),
                ):
                    values[z_of[row[0]]] = row[1:]
                name = f"q={q:.5g}-r={r:.5g}"
                print(
                    f"{name} (shard {s}, serial {serial}, validated {bool(validated)}): "
                    f"R8 {len(values)} nodes (z_samples {z_samples}), "
                    f"z {max(values)!r} .. {min(values)!r}"
                )
                if z_samples != len(values):
                    problems.append(f"{name}: z_samples {z_samples} != {len(values)} values")
                if name not in r3_files:
                    problems.append(f"{name}: no R4 extract to compare with")
                    print("  no R4 extract")
                    continue
                matched.add(name)
                old = {}
                with open(r3_files[name]) as f:
                    for rec in csv.DictReader(f):
                        old[float(rec["z"])] = tuple(
                            float(rec[c]) if rec[c] not in ("", "None") else None
                            for c in VALUE_COLUMNS
                        )
                added = sorted(set(values) - set(old), reverse=True)
                removed = sorted(set(old) - set(values), reverse=True)
                common = set(values) & set(old)
                differing = [
                    z for z in common if not all(same(a, b) for a, b in zip(values[z], old[z]))
                ]
                total_added += len(added)
                ki = {k: continuity[k][:2] for k in continuity}
                print(
                    f"  R4 {len(old)} nodes, z {max(old)!r} .. {min(old)!r}; "
                    f"added {len(added)}, removed {len(removed)}; "
                    f"common {len(common)}, of which differing in any column: {len(differing)}"
                )
                for z in added:
                    notes = []
                    for label, k in (("Tq", q), ("Tr", r)):
                        z_init, z_switch = ki[k]
                        if z_switch <= z < z_init:
                            notes.append(f"{label}: in [z_switch, z_init)")
                        elif z < z_switch:
                            notes.append(f"{label}: below z_switch")
                        else:
                            notes.append(f"{label}: at or above z_init")
                    print(f"    added z = {z!r}  ({'; '.join(notes)})")
                for z in removed:
                    print(f"    removed z = {z!r}")
                for z in sorted(differing, reverse=True)[:10]:
                    print(f"    differs at z = {z!r}: R8 {values[z]} R4 {old[z]}")
    for name in sorted(set(r3_files) - matched):
        problems.append(f"R4 extract {name} has no R8 row")
    print(f"nodes added over all rows: {total_added}")

    print()
    print(f"problems: {len(problems)}")
    for p in problems:
        print(f"  {p}")
    return 0 if not problems and not clamp_hits and n_handover == n_rows else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
