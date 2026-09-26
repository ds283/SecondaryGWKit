"""
Probe: a process killed inside a write transaction on a rollback-journal (journal_mode=delete)
SQLite file leaves a hot journal; the store reader refuses the file; a read-write open rolls it
back. This is what a SIGTERM'd Datastore actor leaves on a shard. No Ray; temporary files under
this directory.

    PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/hot_journal_probe.py
"""

import os
import sqlite3
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

with tempfile.TemporaryDirectory(dir=HERE, prefix="journal-") as tmp:
    db = Path(tmp) / "shard.sqlite"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE t (serial INTEGER PRIMARY KEY, v TEXT)")
    c.execute("INSERT INTO t VALUES (1, 'committed')")
    c.commit()
    c.close()

    pid = os.fork()
    if pid == 0:
        # the child: a write transaction large enough to spill to the file, then killed
        c = sqlite3.connect(db)
        c.execute("PRAGMA cache_size = 1")
        c.execute("BEGIN IMMEDIATE")
        for i in range(2, 5000):
            c.execute("INSERT INTO t VALUES (?, ?)", (i, "x" * 200))
        os._exit(0)  # no commit, no close: what SIGTERM does to an actor
    os.waitpid(pid, 0)

    journal = Path(str(db) + "-journal")
    print(
        "journal left beside the shard:",
        journal.exists(),
        journal.stat().st_size if journal.exists() else 0,
    )
    ro = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
    try:
        print("read-only count:", ro.execute("SELECT count(*) FROM t").fetchone())
    except sqlite3.Error as e:
        print("read-only open:", type(e).__name__, e)
    ro.close()
    rw = sqlite3.connect(db)
    print(
        "read-write count (rolls the journal back):",
        rw.execute("SELECT count(*) FROM t").fetchone(),
    )
    rw.close()
    print("journal after read-write open:", journal.exists())
