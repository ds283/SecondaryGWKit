"""Edit the copied capture harness (07b's on-disk copy, state/capture_harness_r31_sha256.txt) for R32, each edit an
asserted single replacement marked "prompt 07c r32": prov.py -- one tree h, the worktree at cc3368ce (code 67b5af74),
reading R32 through state/stores/r32; worker.py -- each summarised call gains ``chase_07c``, the result dict's
num_chase_stops, num_chase_stopped_leaves and num_chase_noise_bisections (prompt 07c section 1.4), beside 07b's
refusals_07b. Run once: python -I -B edit_harness_07c.py"""
D = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/rehearsal/scripts/capture/"


def edit(name, reps):
    p = D + name
    s = open(p).read()
    for a, b in reps:
        assert s.count(a) == 1, (name, a[:80], s.count(a))
        s = s.replace(a, b)
    open(p, "w").write(s)


edit("prov.py", [
    ('"""[prompt 07b section 5 rehearsal agent, R31:',
     '"""[prompt 07c section 5 rehearsal agent, R32: 07b\'s rehearsal copy (1c976bf8), changed where marked "prompt 07c r32": one tree, h (cc3368ce, code 67b5af74), reading R32; the parents assertion unchanged (the same depth).]\n\n[prompt 07b section 5 rehearsal agent, R31:'),
    ('STORE_DIR_BY_TREE = {"h": MAIN / "var" / "levin-phase-floor-07b" / "rehearsal" / "state" / "stores" / "r31"}  # prompt 07b r31\n',
     'STORE_DIR_BY_TREE = {"h": MAIN / "var" / "levin-phase-floor-07b" / "rehearsal" / "state" / "stores" / "r31"}  # prompt 07b r31\n'
     '# prompt 07c r32: R31\'s tree and store are kept above, by name, for the record; one tree, h, the worktree at Part C\'s\n'
     '# dispatch SHA cc3368ce (code 67b5af74, R32\'s tree), reading R32 through a directory of symbolic links in\n'
     '# var/levin-phase-floor-07c/rehearsal/state/stores/r32 (as R31\'s), opened read-only\n'
     'H07B_TREE, H07B_SHA = H_TREE, H_SHA  # prompt 07c r32: not used\n'
     'H_TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c")  # prompt 07c r32\n'
     'H_SHA = "cc3368ce69d42fd6871d6264fd30efa65dd00cea"  # prompt 07c r32 (code 67b5af74)\n'
     'TREES = {"h": (H_TREE, H_SHA)}  # prompt 07c r32\n'
     'STORE_DIR_BY_TREE = {"h": MAIN / "var" / "levin-phase-floor-07c" / "rehearsal" / "state" / "stores" / "r32"}  # prompt 07c r32\n'),
])
edit("worker.py", [
    ('"""[prompt 07b section 5 rehearsal agent, R31:',
     '"""[prompt 07c section 5 rehearsal agent, R32: 07b\'s rehearsal copy (2a253652), changed where marked "prompt 07c r32": each summarised call (group and other) gains ``chase_07c``, the result dict\'s num_chase_stops, num_chase_stopped_leaves and num_chase_noise_bisections (prompt 07c section 1.4), read after the timed section, beside refusals_07b. Nothing else changes.]\n\n[prompt 07b section 5 rehearsal agent, R31:'),
    ('                           for k_ in ("num_edge_rule_refusals", "num_range_rule_refusals")}\n    return out\n',
     '                           for k_ in ("num_edge_rule_refusals", "num_range_rule_refusals")}\n'
     '    # prompt 07c r32: the chase stop\'s three counts (prompt 07c section 1.4), from the call\'s own result dict (None where absent)\n'
     '    out["chase_07c"] = {k_: (int(res_[k_]) if k_ in res_ else None)\n'
     '                        for k_ in ("num_chase_stops", "num_chase_stopped_leaves", "num_chase_noise_bisections")}\n'
     '    return out\n'),
])
print("edited")
