"""levin-phase-floor prompt 06 section 6, route (B) (the user's decision, 2026-10-26, relayed by the
laptop session): build the unregistered scratch copy of A3 v2 that ``s`` opens read-only.

At the code commit the factories declare ``theta_residual`` on ``GkWKBValue``, ``TkWKBValue`` and
``GkSourceValue``; datastorekit refuses to open a store whose tables lack a declared column, so
``s`` cannot open A3 v2 itself (out/cmd/06_dev_smoke_s.txt). The copy is A3 v2's primary and four
shards, copied byte for byte under their own names (the primary names its shards by file name) into
``state/a3v2_copy/``, with ``ALTER TABLE <t> ADD COLUMN theta_residual FLOAT`` on each of the three
value tables of each shard copy: nullable, and NULL in every row. The sidecar manifest is not
copied: the copy is not a registered store.

A3 v2 is only read here, ``open(..., "rb")``. Every write is to a path under ``state/a3v2_copy/``,
checked before the write. Certification is certify_copy.py.
"""
import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import STORE_DIR, WIDER, sha256  # noqa: E402

COPY_DIR = WIDER / "state" / "a3v2_copy"
NAMES = ["handover-A3-v2-lambdacdm.sqlite"] + [
    f"handover-A3-v2-lambdacdm-shard{i:04d}.sqlite" for i in range(4)
]
TABLES = ["GkWKBValue", "TkWKBValue", "GkSourceValue"]


def main():
    COPY_DIR.mkdir(parents=True, exist_ok=True)
    if any(COPY_DIR.iterdir()):
        raise SystemExit(f"{COPY_DIR} is not empty")
    for name in NAMES:
        src = STORE_DIR / name
        dst = COPY_DIR / name
        h_src = sha256(src)
        with open(src, "rb") as fi, open(dst, "xb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        h_dst = sha256(dst)
        print(f"copied {name}: source sha256 {h_src}, copy sha256 {h_dst}, equal {h_src == h_dst}")
        if h_src != h_dst:
            raise SystemExit("copy differs")
    for i in range(4):
        dst = (COPY_DIR / f"handover-A3-v2-lambdacdm-shard{i:04d}.sqlite").resolve()
        assert dst.parent == COPY_DIR.resolve(), dst
        conn = sqlite3.connect(str(dst))
        try:
            for t in TABLES:
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
                if "theta_residual" in cols:
                    raise SystemExit(f"{dst.name}: {t} already has theta_residual")
                conn.execute(f'ALTER TABLE "{t}" ADD COLUMN theta_residual FLOAT')
                print(f"{dst.name}: ALTER TABLE {t} ADD COLUMN theta_residual FLOAT")
            conn.commit()
        finally:
            conn.close()
    for p in sorted(COPY_DIR.iterdir()):
        print(f"{p.name}  {p.stat().st_size}  sha256 {sha256(p)}")


if __name__ == "__main__":
    main()
