"""D10-4 (prompt 10 §2.5): prompt 02's check at open, on plain copies of a real closed store.

    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/fault_copies.py \\
        var/datastores/datastore-integrity-10-resumed.sqlite <work dir outside var/>

1. Copies the primary and its shards (`shutil.copy2`) into `<work dir>/untouched/` and runs the
   census (`docs/datastore-integrity-audit/replicated_serial_census.py`) on it.
2. Makes one copy per case and changes it with the `sqlite3` command-line tool alone; each
   statement is printed as it is run:
   1. a `redshift` row deleted from one shard;
   2. a `BackgroundModelValue` row duplicated on one shard;
   3. one of the primary's `shard_keys` pointed at a serial that is no `wavenumber`;
   4. a `redshift` row deleted from one replica, with the `replication_in_flight` record that an
      interrupted `get` of it would have left, naming another shard as controller and the row's
      own timestamp as the start.
3. Opens each copy with `extract_TkWKB_data.py --ray-address local`, whose `ShardedPool` open runs
   the check before any actor exists, and prints the opener's exit status and every line of its
   output that the check printed or that names the refusal.
4. Runs the census on the repaired copy, and reads its `replication_in_flight` table.

It never touches the store it copies; the work directory is the caller's to delete.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPO_ROOT / "venv" / "bin" / "python"
CENSUS = (
    REPO_ROOT / "docs" / "datastore-integrity-audit" / "replicated_serial_census.py"
)
OPENER = "extract_TkWKB_data.py"

SHARD_A = 1  # the shard a row is deleted from, in case 1
CONTROLLER = 0  # case 4's controlling shard
REPLICA = 2  # case 4's replica, which lacks the row


def files_of(primary: Path):
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    names = [r[0] for r in con.execute("select filename from shards order by serial")]
    con.close()
    return [primary] + [primary.parent / name for name in names]


def copy_store(primary: Path, into: Path) -> Path:
    into.mkdir(parents=True)
    for f in files_of(primary):
        shutil.copy2(f, into / f.name)
    return into / primary.name


def shard(primary: Path, sid: int) -> Path:
    return files_of(primary)[1 + sid]


def sql(path: Path, statement: str) -> str:
    print(f"    $ sqlite3 {path.parent.name}/{path.name} {statement!r}")
    out = subprocess.run(
        ["sqlite3", str(path), statement], capture_output=True, text=True, check=True
    ).stdout.strip()
    if out:
        print(f"      {out}")
    return out


def census(primary: Path) -> None:
    out = subprocess.run(
        [str(PYTHON), str(CENSUS), str(primary)],
        capture_output=True,
        text=True,
        check=False,
    )
    print("    census:")
    for line in out.stdout.splitlines():
        print(f"      {line}")


WANTED = (
    "ReplicatedDivergence",
    "Replicat",
    "shard key",
    "shard_key",
    "Repaired",
    "repaired",
    "copied",
    "replication_in_flight",
    "Rolled back",
    "Opened existing",
    "RUNNING PIPELINE FOR MODEL",
    "Error",
    "refus",
)


def open_with_reader(primary: Path, out_dir: Path) -> int:
    proc = subprocess.run(
        [
            str(PYTHON),
            "-u",
            OPENER,
            "--database",
            str(primary),
            "--ray-address",
            "local",
            "--output",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    text = proc.stdout + proc.stderr
    (out_dir.parent / f"{out_dir.name}.log").write_text(text)
    print(f"    opener: {OPENER} --ray-address local, exit status {proc.returncode}")
    for line in text.splitlines():
        if "file_system_monitor" in line:
            continue
        if any(w in line for w in WANTED):
            print(f"      | {line}")
    return proc.returncode


def main(source, work) -> int:
    source, work = Path(source).resolve(), Path(work).resolve()
    if work.is_relative_to(REPO_ROOT / "var"):
        raise RuntimeError(f"{work} is under var/; the copies go outside it")
    work.mkdir(parents=True, exist_ok=False)

    print("== the untouched copy")
    untouched = copy_store(source, work / "untouched")
    census(untouched)

    print("== case 1: a redshift row missing on one shard")
    p = copy_store(source, work / "case1")
    s = shard(p, SHARD_A)
    serial = sql(s, "select max(serial) from redshift;")
    sql(s, f"delete from redshift where serial = {serial};")
    open_with_reader(p, work / "case1-out")

    print("== case 2: a BackgroundModelValue row duplicated on one shard")
    p = copy_store(source, work / "case2")
    s = shard(p, SHARD_A)
    cols = [
        r[1]
        for r in sqlite3.connect(f"file:{s}?mode=ro", uri=True).execute(
            'PRAGMA table_info("BackgroundModelValue")'
        )
        if r[1] != "serial"
    ]
    names = ", ".join(f'"{c}"' for c in cols)
    sql(
        s,
        f'insert into "BackgroundModelValue" ({names}) select {names} from '
        f'"BackgroundModelValue" where serial = (select min(serial) from "BackgroundModelValue");',
    )
    open_with_reader(p, work / "case2-out")

    print("== case 3: a shard key that is not a wavenumber")
    p = copy_store(source, work / "case3")
    sql(p, "select key_serial, shard_id from shard_keys order by key_serial;")
    s0 = shard(p, 0)
    bogus = int(sql(s0, "select max(serial) + 1000 from wavenumber;"))
    sql(
        p,
        f"update shard_keys set key_serial = {bogus} where key_serial = "
        f"(select min(key_serial) from shard_keys);",
    )
    open_with_reader(p, work / "case3-out")

    print("== case 4: the repair of an interrupted get")
    p = copy_store(source, work / "case4")
    c, r = shard(p, CONTROLLER), shard(p, REPLICA)
    serial = sql(c, "select max(serial) from redshift;")
    started = sql(c, f"select timestamp from redshift where serial = {serial};")
    sql(r, f"delete from redshift where serial = {serial};")
    sql(p, "select count(*) from replication_in_flight;")
    sql(
        p,
        "insert into replication_in_flight (operation, class_name, controller_shard, store_id, "
        f"started) values ('get', 'redshift', {CONTROLLER}, NULL, '{started}');",
    )
    sql(p, "select * from replication_in_flight;")
    open_with_reader(p, work / "case4-out")
    print("    after the open:")
    sql(p, "select count(*) from replication_in_flight;")
    sql(r, f"select serial, z, timestamp from redshift where serial = {serial};")
    census(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
