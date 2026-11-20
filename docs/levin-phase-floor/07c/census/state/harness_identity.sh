#!/bin/bash
# levin-phase-floor 07c section 6 (census agent): the harness is 07b's census harness, byte-identical to the copy that ran on the Mac Studio.
set -u
D=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c/docs/levin-phase-floor/07b/census/scripts
V=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/census/scripts
C=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/census/scripts
echo "== diff -r (docs in the tree) (07b's Mac Studio copy)"; diff -r "$D" "$V"; echo "rc $?"
echo "== diff -r (docs in the tree) (07c's copy, before any change)"; diff -r "$D" "$C"; echo "rc $?"
echo "== file counts"; (cd "$D" && find . -type f | wc -l); (cd "$V" && find . -type f | wc -l); (cd "$C" && find . -type f | wc -l)
echo "== sha256 of every file, 07b's Mac Studio copy"; (cd "$V" && find . -type f | LC_ALL=C sort | xargs shasum -a 256)
echo "== sha256 of the list, each copy"
for x in "$D" "$V" "$C"; do (cd "$x" && find . -type f | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | sed "s|-|$x|"); done
