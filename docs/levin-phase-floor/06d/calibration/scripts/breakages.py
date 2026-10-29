import subprocess, shutil, sys
P = "AdaptiveLevin/levin_quadrature.py"
CLEAN = sys.argv[1]
orig = open(CLEAN).read()
B = {
 "(i) floor back to eq. (151)": ("    endpoint_proxy = 0.5 * f_scale * width\n",
   "    _G0, _G1 = _levin_G0_G1(theta_prime_Cheb, width)\n    round_off_err = _roundoff_floor(f_scale, width, _G0, _G1, chebyshev_order)\n    endpoint_proxy = 0.5 * f_scale * width\n"),
 "(ii) argument term removed": ("        + _UNIT_ROUNDOFF * argument_term\n", "        + 0.0 * argument_term\n"),
 "(iii) guard removed": ("    if current == 0.0 or not np.isfinite(current):\n        return None\n", ""),
 "(iv) node term removed": ("        + node_term\n", "        + 0.0 * node_term\n"),
}
try:
    for name, (a, b) in B.items():
        assert orig.count(a) == 1, name
        open(P, "w").write(orig.replace(a, b))
        r = subprocess.run([ "./venv/bin/python", "-m", "unittest", "AdaptiveLevin.tests.test_levin_cc_floor"], capture_output=True, text=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"})
        fails = sorted({l.split("(")[0].replace("FAIL: ","").replace("ERROR: ","").strip() + " " + (l[l.index("(problem") :] if "(problem" in l else (l[l.index(") (")+2:] if ") (" in l else "")) for l in r.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))})
        print("==", name); print("  ", r.stderr.strip().splitlines()[-1])
        for f in fails: print("   ", f)
        errs = [l for l in r.stderr.splitlines() if l.startswith(("AssertionError", "ZeroDivisionError"))]
        for e in errs[:12]: print("      ", e[:220])
finally:
    open(P, "w").write(orig)
