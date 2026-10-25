#!/bin/bash
# The Gate of levin-phase-floor prompt 01, read-only. Run from the repository root:
#   bash var/levin-phase-floor-01/scripts/gate_01.sh > var/levin-phase-floor-01/out/gate_01.txt 2>&1
DISPATCH=757ef016fc37be996c342d162c05f17b3c430e4d
PRIMARY=var/datastores/handover-A3-v2-lambdacdm.sqlite
echo "# gate_01.sh, run $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
echo "# script sha256 $(shasum -a 256 "$0" | cut -d' ' -f1)"
echo
echo "== 1. dispatch commit pushed, and this prompt's commit or later (read-only: no fetch)"
echo "\$ git rev-parse HEAD"; git rev-parse HEAD
echo "\$ git rev-parse origin/handover-remedial   (as last fetched)"; git rev-parse origin/handover-remedial
echo "\$ git rev-parse --abbrev-ref HEAD"; git rev-parse --abbrev-ref HEAD
echo "HEAD == dispatch SHA: $([ "$(git rev-parse HEAD)" = "$DISPATCH" ] && echo yes || echo NO)"
git merge-base --is-ancestor "$DISPATCH" origin/handover-remedial && echo "dispatch SHA is an ancestor of (or equal to) origin/handover-remedial: yes" || echo "dispatch SHA is an ancestor of origin/handover-remedial: NO"
PROMPT_COMMIT=$(git log -1 --format=%H -- prompts/levin-phase-floor/01-phase-group-census.md)
echo "last commit touching the prompt file: $PROMPT_COMMIT"
git merge-base --is-ancestor "$PROMPT_COMMIT" "$DISPATCH" && echo "prompt's commit is an ancestor of (or equal to) the dispatch SHA: yes" || echo "prompt's commit is an ancestor of the dispatch SHA: NO"
echo
echo "== 2. runregistrykit list"
echo "\$ ./venv/bin/python -m runregistrykit list"; ./venv/bin/python -m runregistrykit list
echo
echo "== 3. git status --short (empty expected; var/ is gitignored)"
echo "\$ git status --short"; git status --short; echo "(end of git status; lines: $(git status --short | wc -l | tr -d ' '))"
echo
echo "== 4. store show"
echo "\$ ./venv/bin/python -m runregistrykit store show $PRIMARY"; ./venv/bin/python -m runregistrykit store show "$PRIMARY"
