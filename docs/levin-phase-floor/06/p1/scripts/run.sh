#!/bin/sh
# p1: run one p1 script under one tree, recording the command, the scripts' SHA-256 and all output
# in out/<logname>.log. Usage: run.sh TREE LOGNAME SCRIPT [ARGS...]   (TREE: r13 | r14 | none)
set -u
P1=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06/p1
PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
TREE=$1; LOGNAME=$2; SCRIPT=$3; shift 3
case $TREE in
  r13) PP=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06-9f66622 ;;
  r14) PP=/Users/ds283/Documents/Code/SecondaryGWKit-lpf06-f5cee4c ;;
  none) PP= ;;
  *) echo "bad tree $TREE"; exit 2 ;;
esac
LOG=$P1/out/$LOGNAME.log
cd $P1/out || exit 2
{
  echo "# command: PYTHONPATH=$PP OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $PY -B $P1/scripts/$SCRIPT $*"
  echo "# cwd: $(pwd); host $(hostname); start $(date -u +%FT%TZ)"
  (cd $P1/scripts && shasum -a 256 *.py *.sh | sed 's/^/# script sha256 /')
} > $LOG
PYTHONPATH=$PP OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $PY -B $P1/scripts/$SCRIPT "$@" >> $LOG 2>&1
RC=$?
echo "# exit $RC at $(date -u +%FT%TZ)" >> $LOG
exit $RC
