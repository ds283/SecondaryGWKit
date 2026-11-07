"""levin-phase-floor 07 section 0.5: the earlier attempts' records, copied byte for byte into
docs/levin-phase-floor/07/earlier-attempts/, checked against the set-aside tarball. New in 07.

The tarball var/levin-phase-floor-07/wip/07-part-a-rebased-on-06d-untracked.tar.gz (read only; its own SHA-256
checked against that directory's SHA256SUMS.txt) holds docs/levin-phase-floor/07/calibration/ (stops 1-3 under
stop1/-stop3/, and the fourth stop's scripts/ and out/) and prompts/levin-phase-floor/logs/
07-stationary-point-guard.md. They were copied to earlier-attempts/calibration/ and, the log renamed,
earlier-attempts/log-parts-a-1-to-4.md. This script reads every regular file of the tarball with Python's
tarfile, hashes it, hashes its copy, and writes the list, one line per file (tarball sha256, copy sha256,
equal, member, copy), to earlier-attempts/SHA256_CHECK.txt. It also lists any file under earlier-attempts/ that
is not a copy (the check file itself aside). The tarball's macOS AppleDouble entries ("._" names, extended
attributes stored by tar beside each entry) are listed and skipped: they are not records. Exit 1 if any file
differs or is missing.
"""

import hashlib
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

TARBALL = (
    prov.REPO
    / "var"
    / "levin-phase-floor-07"
    / "wip"
    / "07-part-a-rebased-on-06d-untracked.tar.gz"
)
SUMS = TARBALL.parent / "SHA256SUMS.txt"
DEST = prov.REPO / "docs" / "levin-phase-floor" / "07" / "earlier-attempts"
CHECK = DEST / "SHA256_CHECK.txt"


def copy_path(member: str) -> Path:
    prefix = "docs/levin-phase-floor/07/"
    if member.startswith(prefix):
        return DEST / member[len(prefix) :]
    if member == "prompts/levin-phase-floor/logs/07-stationary-point-guard.md":
        return DEST / "log-parts-a-1-to-4.md"
    raise ValueError(member)


def main():
    tar_sha = prov.sha256(TARBALL)
    listed = any(
        line.split()[:2] == [tar_sha, TARBALL.name]
        for line in SUMS.read_text().splitlines()
        if line.strip()
    )
    lines = prov.header(
        __file__,
        "section 0.5, the earlier attempts' records against the tarball",
        (TARBALL,),
    )
    lines.append(
        f"# tarball sha256 {tar_sha}; listed in {SUMS.relative_to(prov.REPO)}: {listed}"
    )
    bad = 0
    seen = set()
    with tarfile.open(TARBALL, "r:gz") as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            if not member.isfile():
                continue
            if Path(member.name).name.startswith("._"):
                # macOS AppleDouble metadata (extended attributes) that tar stored beside each entry;
                # bsdtar restores it as attributes, not as a file, so it is not a record and has no copy
                lines.append(
                    f"# AppleDouble metadata, not a record, not copied: {member.name}"
                )
                continue
            want = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
            dest = copy_path(member.name)
            seen.add(dest)
            got = prov.sha256(dest) if dest.is_file() else "MISSING"
            equal = got == want
            bad += not equal
            lines.append(
                f"{want}  {got}  {'equal' if equal else 'DIFFERENT'}  {member.name}  "
                f"{dest.relative_to(prov.REPO)}"
            )
    extra = sorted(
        p for p in DEST.rglob("*") if p.is_file() and p not in seen and p != CHECK
    )
    for p in extra:
        lines.append(f"# not a copy: {p.relative_to(prov.REPO)}")
        bad += 1
    lines.append(
        f"# {len(seen)} files; {bad} differing, missing or extra; tarball listed: {listed}"
    )
    CHECK.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    if bad or not listed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
