"""levin-phase-floor prompt 06m, section 2: the deliberate breakages (i)-(v), each applied to
AdaptiveLevin/levin_quadrature.py's _trust_noise_yardstick() by exact text replacements, the whole of
AdaptiveLevin/tests/test_levin_trust_yardstick.py run in a subprocess (so that every test that catches it is
recorded), then reverted. The module's SHA-256 is checked equal to the clean file's after every revert and at the
end. 06g's breakages.py (docs/levin-phase-floor/06l/scripts/calibration/breakages.py, 06l's copy) is the pattern;
the breakages and the test are 06m's.

  (i)   |E_L| + |E_R| in place of |E_L + E_R|: test 1 must fail;
  (ii)  a Clenshaw-Curtis half's phi^- counted on a Levin parent: tests 1 and 4 must fail;
  (iii) the |sum E| term dropped: tests 1 and 3 must fail;
  (iv)  a Clenshaw-Curtis half's full phi_h in place of phi^-_h: test 1 must fail;
  (v)   the |sum E| term added before the phi^- terms: test 1 must fail, and test 2's comparison with the
        variant is run and its result recorded.

Run from the repository root, with the module clean: PYTHONPATH=. ./venv/bin/python -B <this script>. Writes
../out/breakages.txt. No Ray, no datastore.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

OUT = HERE.parent / "out" / "breakages.txt"
LEVIN = prov.LEVIN
TEST_FILE = prov.REPO / "AdaptiveLevin" / "tests" / "test_levin_trust_yardstick.py"
TEST = "AdaptiveLevin.tests.test_levin_trust_yardstick"

# each test method's number in prompt 06m section 2
NUMBER = {
    "test_cases_bit_for_bit": 1,
    "test_order_matters_and_is_kept": 1,
    "test_signed_sum_is_below_the_absolute_sum": 1,
    "test_levin_half_without_a_floor_reads_zero": 1,
    "test_raises_on_a_cc_parent_whose_cc_half_lacks_its_parts": 1,
    "test_bit_identical_to_the_variant": 2,
    "test_trusted_on_cancelling_node_errors": 3,
    "test_levin_parent_keeps_cc_halves_out": 4,
    "test_levin_over_levin_far_above_noise_stays_untrusted": 5,
}

_SIGNED = (
    "            node_sum = (\n"
    '                cc_halves[0]["node_error_signed"] + cc_halves[1]["node_error_signed"]\n'
    "            )\n"
)
_CC_LOOP = (
    "        for half in cc_halves:\n"
    "            if (\n"
    '                half.get("phase_err_less_node") is None\n'
    '                or half.get("node_error_signed") is None\n'
    "            ):\n"
    "                raise RuntimeError(\n"
    '                    "levin_quadrature: a Clenshaw-Curtis half of a Clenshaw-Curtis parent has no "\n'
    '                    "phase_err_less_node or node_error_signed (_trust_noise_yardstick())"\n'
    "                )\n"
    '            nu = nu + half["phase_err_less_node"]\n'
)
_NODE_SUM = (
    "        if len(cc_halves) == 1:\n"
    '            node_sum = cc_halves[0]["node_error_signed"]\n'
    "        else:\n" + _SIGNED
)
_ADD_S = "        nu = nu + abs(node_sum)\n"

BREAKAGES = [
    (
        "(i) |E_L| + |E_R| in place of |E_L + E_R|",
        [
            (
                _SIGNED,
                "            node_sum = abs(cc_halves[0][\"node_error_signed\"]) + abs(\n"
                '                cc_halves[1]["node_error_signed"]\n'
                "            )\n",
            )
        ],
        {1},
    ),
    (
        "(ii) a Clenshaw-Curtis half's phi^- counted on a Levin parent",
        [
            (
                "    if parent_is_direct and any(half_is_direct):\n",
                "    if not parent_is_direct:\n"
                "        for half, is_direct in zip(halves, half_is_direct):\n"
                "            if is_direct:\n"
                '                nu = nu + half["phase_err_less_node"]\n'
                "    if parent_is_direct and any(half_is_direct):\n",
            )
        ],
        {1, 4},
    ),
    (
        "(iii) the |sum E| term dropped",
        [(_ADD_S, "        nu = nu + 0.0 * 0.0\n")],
        {1, 3},
    ),
    (
        "(iv) a Clenshaw-Curtis half's full phi_h in place of phi^-_h",
        [
            (
                '            nu = nu + half["phase_err_less_node"]\n',
                '            nu = nu + half["phase_err"]\n',
            )
        ],
        {1},
    ),
    (
        "(v) the |sum E| term added before the phi^- terms",
        [(_CC_LOOP + _NODE_SUM + _ADD_S, _NODE_SUM + _ADD_S + _CC_LOOP)],
        {1},
    ),
]

_LINE = re.compile(r"^(test_\w+) \(([\w.]+)\)")


def outcomes(stderr):
    """Each test method's outcome from unittest -v's stderr: ok, FAIL or ERROR."""
    found = {}
    pending = None
    for line in stderr.split("\n"):
        m = _LINE.match(line)
        if m:
            pending = m.group(1)
        if pending is not None:
            for tail, outcome in ((" ... ok", "ok"), (" ... FAIL", "FAIL"), (" ... ERROR", "ERROR")):
                if line.endswith(tail):
                    found[pending] = outcome
                    pending = None
                    break
    return found


def main():
    clean = LEVIN.read_text()
    clean_sha = prov.sha256(LEVIN)
    lines = prov.header(
        __file__,
        "breakages (i)-(v), each applied, the test module run, then reverted",
        inputs=(LEVIN, TEST_FILE),
    )
    lines.append(f"# clean AdaptiveLevin/levin_quadrature.py sha256 {clean_sha}")
    env = dict(os.environ, PYTHONPATH=".")
    verdicts = []
    try:
        for name, replacements, required in BREAKAGES:
            broken = clean
            for old, new in replacements:
                assert broken.count(old) == 1, (name, old)
                broken = broken.replace(old, new)
            LEVIN.write_text(broken)
            try:
                result = subprocess.run(
                    [sys.executable, "-B", "-m", "unittest", "-v", TEST],
                    cwd=prov.REPO,
                    env=env,
                    capture_output=True,
                    text=True,
                )
            finally:
                LEVIN.write_text(clean)
            assert prov.sha256(LEVIN) == clean_sha
            found = outcomes(result.stderr)
            failing = sorted(t for t, o in found.items() if o != "ok")
            caught_by = sorted({NUMBER[t] for t in failing})
            lines.append("")
            lines.append(f"== {name}: {len(replacements)} replacement(s); exit {result.returncode}")
            for t in sorted(found, key=lambda t: (NUMBER.get(t, 0), t)):
                lines.append(f"   test {NUMBER.get(t, '?')} {t}: {found[t]}")
            stderr = result.stderr.split("\n")
            lines += [f"   {line}" for line in stderr if line.startswith(("Ran ", "OK", "FAILED"))]
            lines += [
                f"   | {line.strip()}"
                for line in stderr
                if line.strip().startswith(("AssertionError", "RuntimeError", "KeyError", "TypeError"))
            ][:12]
            ok = required <= set(caught_by)
            lines.append(
                f"   required {sorted(required)}; failing tests {caught_by}; {'CAUGHT' if ok else 'NOT CAUGHT'}"
            )
            if name.startswith("(v)"):
                lines.append(
                    f"   test 2 (the comparison with the variant) under (v):"
                    f" {found.get('test_bit_identical_to_the_variant')}"
                )
            verdicts.append((name, ok, caught_by))
    finally:
        LEVIN.write_text(clean)
        assert prov.sha256(LEVIN) == clean_sha
    lines.append("")
    for name, ok, caught_by in verdicts:
        lines.append(f"{name}: {'CAUGHT' if ok else 'NOT CAUGHT'} (failing tests {caught_by})")
    lines.append(f"module restored: {prov.sha256(LEVIN) == clean_sha} (sha256 {prov.sha256(LEVIN)})")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
