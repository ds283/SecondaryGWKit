#!/bin/bash
# levin-phase-floor 07b section 6 (census agent): the harness is 07a's census harness, byte-identical to the copy that ran on the Mac Studio.
set -u
D=/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5/docs/levin-phase-floor/07a/census/scripts
V=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/census/scripts
C=/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/census/scripts
echo "== diff -r (docs in tree h) (07a's Mac Studio copy)"; diff -r "$D" "$V"; echo "rc $?"
echo "== diff -r (docs in tree h) (07b's copy, before any change)"; diff -r "$D" "$C"; echo "rc $?"
echo "== file counts"; (cd "$D" && find . -type f | wc -l); (cd "$V" && find . -type f | wc -l); (cd "$C" && find . -type f | wc -l)
echo "== sha256 of every file, 07a's Mac Studio copy"; (cd "$V" && find . -type f | LC_ALL=C sort | xargs shasum -a 256)
echo "== sha256 of the list, each copy"
for x in "$D" "$V" "$C"; do (cd "$x" && find . -type f | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | sed "s|-|$x|"); done
