"""R4's logs against R3's, in the shape of docs/rrk-v020-adoption/rehearsal/log_comparison_03.txt.

Run from the repository root:  ./venv/bin/python var/handover-06/scripts/log_comparison.py
R3's raw stdout.log/stderr.log are here only for `reference` and `resumed-c`
(var/handover-06/recorded/); for every slug R3's counts are read from log_comparison_03.txt.
"""

import glob
import json
import os
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")
R3_TXT = "docs/rrk-v020-adoption/rehearsal/log_comparison_03.txt"
R3_DIRS = {
    "reference": "var/handover-06/recorded/rrk-v020-adoption-03-reference-20261018T232838",
    "resumed-c": "var/handover-06/recorded/rrk-v020-adoption-03-resumed-c-20261018T233941",
}
SLUGS = ["reference", "resumed-a", "resumed-b", "resumed-c"]
SEARCHES = [
    ("replaced 1 occurrence", r"replaced 1 occurrence", 0),
    ("rehearsal scope banner", r"rehearsal scope, NOT A3 v2", 0),
    ("main.py argv", r"main\.py argv", 0),
    ("--no-prune-unvalidated", r"--no-prune-unvalidated", 0),
    ("bootstrapping a local Ray instance", r"bootstrapping a local Ray instance", 0),
    ("Started a local Ray instance", r"Started a local Ray instance", 0),
    (
        "reconcil|repair|in.flight|replicat (i)",
        r"reconcil|repair|in.flight|replicat",
        re.I,
    ),
    ("Traceback", r"Traceback", 0),
    (
        "ModuleNotFoundError|ImportError|AttributeError",
        r"ModuleNotFoundError|ImportError|AttributeError",
        0,
    ),
    ("pickle/unpickl", r"pickle|unpickl", 0),
    ("Datastore.SQL|Datastore/SQL", r"Datastore\.SQL|Datastore/SQL", 0),
    (
        "ray.kill|killed actor|ActorDiedError|RayActorError",
        r"ray\.kill|killed actor|ActorDiedError|RayActorError",
        0,
    ),
    ("ValueError", r"ValueError", 0),
    ("RuntimeWarning", r"RuntimeWarning", 0),
    ("did not validate", r"did not validate", 0),
    ("DeprecationWarning", r"DeprecationWarning", 0),
    ("_timestamp", r"_timestamp", 0),
    ("ConfigError", r"ConfigError", 0),
    ("RunRegistry (not Kit)", r"RunRegistry(?!Kit)", 0),
    ("datastorekit", r"datastorekit", 0),
    ("runregistrykit", r"runregistrykit", 0),
    ("INTEGRITY WARNING", r"INTEGRITY WARNING", 0),
    ("file_system_monitor (disk space)", r"file_system_monitor|is over 95% full", 0),
    ("Rolled back the journal", r"Rolled back the journal", 0),
    ("SystemExit(1)", r"SystemExit\(1\)", 0),
    ("SystemExit(143)", r"SystemExit\(143\)", 0),
    ("IntegrationWarning", r"IntegrationWarning", 0),
    ("Killed: 9", r"Killed: 9", 0),
    ("resuming into", r"resuming into", 0),
]


def lines(path):
    if not os.path.exists(path):
        return None
    return [ANSI.sub("", l) for l in open(path, errors="replace").read().splitlines()]


def norm(line):
    l = ANSI.sub("", line)
    l = l.replace("/Users/ds283/Documents/Code/SecondaryGWKit", "<repo>")
    l = re.sub(r"rrk-v020-adoption-03|handover-06", "<run>", l)
    l = re.sub(r"rrk-v020-adoption|handover", "<campaign>", l)
    l = re.sub(r"--prompt (03|06)", "--prompt <p>", l)
    l = re.sub(
        r"docs/handover/quadsource_atol_sweep\.py|RunDrivers/quadsource_atol_sweep\.py",
        "<launcher>",
        l,
    )
    l = re.sub(r"--purpose .*?(--campaign)", r"--purpose <purpose> \1", l)
    l = re.sub(r"\[repeated \d+x across cluster\]", "[repeated Nx across cluster]", l)
    l = re.sub(r"\d+(\.\d+)?(e[+-]?\d+)?", "N", l)
    return l.strip()


def r3_counts():
    """Section 1 and section 3 of R3's log_comparison_03.txt, by slug."""
    text = open(R3_TXT).read().splitlines()
    s1, s3, slug, sec = {}, {}, None, None
    for l in text:
        if l.startswith("== "):
            sec = l[3]
            continue
        m = re.match(r"-- (\S+) \(", l)
        if m:
            slug = m.group(1)
            continue
        if sec == "1" and slug:
            m = re.match(r"   (.+?)\s+(\d+)\s+(\d+)\s+(\d+)$", l)
            if m:
                s1.setdefault(slug, {})[m.group(1).strip()] = tuple(
                    int(x) for x in m.groups()[1:]
                )
        if sec == "3" and slug:
            m = re.search(r"counts \(.*\) R3 \((.*?)\) R2", l)
            if m:
                s3[slug] = tuple(int(x) for x in m.group(1).split(","))
    return s1, s3


def counts_of(ls):
    return (
        sum(l.startswith("** ") for l in ls),
        sum("ALL WORK ITEMS COMPLETE in time" in l for l in ls),
        sum("Queue summary:" in l for l in ls),
        sum(
            "(no compute items were evaluated - all assets read from datastore)" in l
            for l in ls
        ),
        sum(
            bool(re.search(r"\d+(\.\d+)?% \(\d+ of \d+ work items remaining\)", l))
            for l in ls
        ),
        sum("inflight" in l for l in ls),
    )


def main():
    r3s1, r3s3 = r3_counts()
    run = {
        s: sorted(glob.glob(f"var/runs/handover-06-{s}-*/"))[0].rstrip("/")
        for s in SLUGS
    }
    print(
        "== 1. searches, count per file (stdout.log / stderr.log / launch log), R4 then [R3]; ! marks a difference"
    )
    for s in SLUGS:
        d = run[s]
        files = [
            lines(f"{d}/stdout.log"),
            lines(f"{d}/stderr.log"),
            lines(f"var/runs/handover-06-{s}.launch.log"),
        ]
        print(f"-- {s} ({os.path.basename(d)})")
        for label, pat, flags in SEARCHES:
            c = tuple(sum(bool(re.search(pat, l, flags)) for l in f) for f in files)
            r3 = r3s1.get(s, {}).get(label)
            mark = "" if r3 is None or r3 == c else "  !"
            print(f"   {label:<52} {c[0]:>4} {c[1]:>4} {c[2]:>4}   [R3 {r3}]{mark}")
        for name, f in zip(("stdout.log", "stderr.log", "launch log"), files):
            for l in f:
                if re.search(
                    r"datastorekit|runregistrykit|raypoolkit", l
                ) and not re.search(
                    r"site-packages/(datastorekit|runregistrykit|raypoolkit)", l
                ):
                    print(f"   [{name}] names a package: {l.strip()[:200]}")
    print()
    print(
        "== 2. line kinds against R3's same slug (normalised; set difference; line counts R4/R3); R3's raw logs only for reference and resumed-c"
    )
    for s in SLUGS:
        if s not in R3_DIRS:
            print(
                f"-- {s}: R3's logs are not on this machine; compared by section 1's counts and section 3 only"
            )
            continue
        for f in ("stdout.log", "stderr.log"):
            a, b = lines(f"{run[s]}/{f}"), lines(f"{R3_DIRS[s]}/{f}")
            na, nb = {norm(l) for l in a}, {norm(l) for l in b}
            print(
                f"-- {s} {f}: {len(a)} lines against {len(b)}; R4-only kinds {len(na - nb)}, R3-only kinds {len(nb - na)}"
            )
            for k in sorted(na - nb):
                print(f"   R4 only: {k[:220]}")
            for k in sorted(nb - na):
                print(f"   R3 only: {k[:220]}")
    print()
    print("== 3. '** ' sequences and printed-line counts")
    for s in SLUGS:
        a = lines(f"{run[s]}/stdout.log")
        c = counts_of(a)
        print(
            f"-- {s}: counts (** , complete, Queue summary, no-compute, progress, inflight) R4 {c} R3 {r3s3.get(s)}; equal: {c == r3s3.get(s)}"
        )
        if s in R3_DIRS:
            b = lines(f"{R3_DIRS[s]}/stdout.log")
            sa = [norm(l) for l in a if l.startswith("** ")]
            sb = [norm(l) for l in b if l.startswith("** ")]
            ba = [l for l in a if re.match(r"\*\* (CALCULATE|APPLY|BUILD)", l)]
            bb = [l for l in b if re.match(r"\*\* (CALCULATE|APPLY|BUILD)", l)]
            print(
                f"   '** ' sequence equal to R3's normalised: {sa == sb}; stage banners verbatim equal: {ba == bb} ({len(ba)}/{len(bb)})"
            )
            if sa != sb:
                for x, y in zip(sa, sb):
                    if x != y:
                        print(f"     R4: {x[:200]}\n     R3: {y[:200]}")
        print(
            "   banners: "
            + " | ".join(
                l[3:] for l in a if re.match(r"\*\* (CALCULATE|APPLY|BUILD)", l)
            )
        )
    print()
    print("== 4. status fields at each run's end")
    for s in SLUGS:
        st = json.load(open(f"{run[s]}/status.json"))
        print(
            f"-- {s}: state {st['state']} exit {st['exit_code']} stage {st['stage']!r} units {st['units_done']}/{st['units_total']}; keys {sorted(st)}"
        )
    print()
    print(
        "== 5. stage times (ALL WORK ITEMS COMPLETE), not compared; R3's beside, from its log_comparison_03.txt section 5"
    )
    r3t = {}
    slug = None
    sec = None
    for l in open(R3_TXT).read().splitlines():
        if l.startswith("== "):
            sec = l[3]
            continue
        if sec == "5":
            m = re.match(r"-- (\S+)", l)
            if m:
                slug = m.group(1)
                continue
            if slug and l.strip():
                r3t.setdefault(slug, []).append(l.strip())
    for s in SLUGS:
        a = lines(f"{run[s]}/stdout.log")
        stage = None
        out = []
        for l in a:
            if re.match(r"\*\* (CALCULATE|APPLY)", l):
                stage = l[3:].strip()
            m = re.search(r"ALL WORK ITEMS COMPLETE in time (.*)$", l)
            if m and stage:
                out.append(f"{stage}: {m.group(1).strip()}")
        print(f"-- {s}")
        for x in out:
            print(f"   R4 {x}")
        for x in r3t.get(s, []):
            print(f"   R3 {x}")


if __name__ == "__main__":
    main()
