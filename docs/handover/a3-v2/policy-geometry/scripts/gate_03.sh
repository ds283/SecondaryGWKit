#!/bin/bash
# Gate checks for handover prompt 03. Read-only.
cd /Users/ds283/Documents/Code/SecondaryGWKit
run() { echo; echo "\$ $*"; "$@" 2>&1; echo "[exit $?]"; }
date -u +"gate run (UTC): %Y-%m-%dT%H:%M:%SZ"
run hostname
run git rev-parse HEAD
run git branch --show-current
run git status --short
run git branch -r --contains HEAD
run git rev-parse @{u}
run git merge-base --is-ancestor f1a3888 HEAD
run git diff 98e68d3 HEAD --stat -- ComputeTargets/GkSourcePolicyData.py ComputeTargets/GkSource.py ObjectFactories/GkSourcePolicyData.py ObjectFactories/GkSource.py
run ./venv/bin/python -m runregistrykit list
run ./venv/bin/python -m runregistrykit store show var/datastores/handover-A3-v2-lambdacdm.sqlite
run ./venv/bin/python -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite
run ls -la var/datastores
run ls var
