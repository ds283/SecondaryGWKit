#!/bin/sh
# levin-phase-floor 06i: run a script of this directory at the census tree (6d81fad), as the
# pickles require. Usage: sh run.sh SCRIPT.py [args...]
HERE=$(cd "$(dirname "$0")" && pwd)
MAIN=/Users/ds283/Documents/Code/SecondaryGWKit
TREE=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06i-6d81fad
cd "$TREE" || exit 1
PYTHONPATH=".:$MAIN/var/levin-phase-floor-06e/census/scripts:$HERE" PYTHONDONTWRITEBYTECODE=1 \
  exec "$MAIN/venv/bin/python" -B -u "$HERE/$1" "${@:2}"
