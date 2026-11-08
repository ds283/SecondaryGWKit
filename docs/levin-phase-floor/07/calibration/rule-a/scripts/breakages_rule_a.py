"""levin-phase-floor 07 section 10 (Rule A): the deliberate breakages (x) and (xi), each with the test that must
catch it. New in 07's amendment; the mechanism is 07's scripts/breakages.py's.

Each breakage is a text substitution in a copy of AdaptiveLevin/levin_quadrature.py, checked to occur exactly
once, written to a temporary file outside the repository and installed in place of the module by
variant_loader.py (this directory's copy of 07's), before anything imports AdaptiveLevin, in a fresh process. The
tree is never edited, so nothing has to be reverted. The named test is then run by unittest, and the breakage is
caught if it fails or errors. As a control, every named test is also run once against the unbroken tree.
- **(x)** Rule A off (its threshold 0, so no phase is skipped): test 13 (a) must fail;
- **(xi)** a phase Rule A skipped tested by the guard for "not of one strict sign" rather than for a sign
  change: test 13 (d) must fail.
Writes rule-a/out/breakages_rule_a.json and .txt.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

LQ = prov.REPO / "AdaptiveLevin" / "levin_quadrature.py"
TESTS = prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_stationary.py"
T = "AdaptiveLevin.tests.test_levin_stationary.TestRuleA."

BREAKAGES = {
    "x": (
        "Rule A off (its threshold 0)",
        [("        if variation < SIX_PI:\n", "        if variation < 0.0:\n")],
        [T + "test_13a_zero_rate"],
    ),
    "xi": (
        'a skipped phase tested by the guard for "not of one strict sign"',
        [
            (
                "                if _changes_sign(sample):\n",
                "                if not _strictly_one_sign(sample):\n",
            )
        ],
        [T + "test_13d_several_phases"],
    ),
}

RUNNER = r"""
import json, sys, unittest
sys.path.insert(0, {repo!r})
sys.path.insert(0, {here!r})
path, sha, test = {path!r}, {sha!r}, {test!r}
if path:
    import variant_loader
    variant_loader.install(path, sha, "breakage")
suite = unittest.defaultTestLoader.loadTestsFromName(test)
result = unittest.TestResult()
suite.run(result)
print("RESULT " + json.dumps({{"failures": len(result.failures), "errors": len(result.errors), "run": result.testsRun,
    "detail": [t[1][-800:] for t in result.failures + result.errors][:1]}}))
"""


def run(path, sha, test, timeout=1800):
    code = RUNNER.format(
        repo=str(prov.REPO), here=str(HERE), path=str(path or ""), sha=sha, test=test
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.mkdtemp(prefix="lpf07-rule-a-breakage-"),
        )
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        return {"harness_error": proc.stderr[-1500:]}
    return json.loads(lines[-1][7:])


def caught(r):
    return bool(r.get("failures") or r.get("errors") or r.get("timeout"))


def main():
    out = {
        "header": prov.json_header(
            __file__,
            "section 10 (Rule A), the deliberate breakages (x) and (xi)",
            (LQ, TESTS, HERE / "variant_loader.py"),
        ),
        "breakages": {},
    }
    source = LQ.read_text()
    txt = [f"# {line}" for line in out["header"]] + [""]
    for key, (what, subs, tests) in BREAKAGES.items():
        broken = source
        for old, new in subs:
            count = broken.count(old)
            if count != 1:
                raise SystemExit(f"breakage ({key}): {old!r} occurs {count} times")
            broken = broken.replace(old, new)
        path = Path(tempfile.mkdtemp(prefix="lpf07-rule-a-variant-")) / (
            f"levin_quadrature_breakage_{key}.py"
        )
        path.write_text(broken)
        sha = prov.sha256(path)
        entry = {"what": what, "substitutions": subs, "variant_sha256": sha}
        entry["tests"] = {}
        for test in tests:
            control = run(None, "", test)
            result = run(path, sha, test)
            entry["tests"][test] = {
                "control_unbroken": control,
                "control_passes": not caught(control) and control.get("run") == 1,
                "broken": result,
                "caught": caught(result),
            }
            txt.append(
                f"({key}) {what}: {test.split('.')[-1]}: control "
                f"{'passes' if entry['tests'][test]['control_passes'] else 'FAILS'}; "
                f"broken {'CAUGHT (fails)' if caught(result) else 'NOT CAUGHT (passes)'}"
            )
            if result.get("detail"):
                txt.append("    " + result["detail"][0].strip().splitlines()[-1])
        out["breakages"][key] = entry
    out["all_caught"] = all(
        t["caught"] and t["control_passes"]
        for e in out["breakages"].values()
        for t in e["tests"].values()
    )
    txt.append("")
    txt.append(f"all caught, controls passing: {out['all_caught']}")
    (prov.OUT / "breakages_rule_a.json").write_text(json.dumps(out, indent=1))
    (prov.OUT / "breakages_rule_a.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
