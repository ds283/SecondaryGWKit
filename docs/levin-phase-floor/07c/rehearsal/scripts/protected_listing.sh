#!/bin/sh
# protected_listing.sh OUT -- list (mtime size path) of every file under the paths this agent must not write
# (state/protected_paths.txt) and of A3 v2's, R29's, R30's and R31's store files, sorted by path. Read only.
cd /Users/ds283/Documents/Code/SecondaryGWKit || exit 2
P=$(cat var/levin-phase-floor-07c/rehearsal/state/protected_paths.txt)
{
  command find $P -type f -print0 | xargs -0 stat -c '%Y %s %n'
  ls var/datastores | grep -E "A3-v2|07-r29|07a-r30|07b-r31" | sed 's#^#var/datastores/#' | xargs stat -c '%Y %s %n'
} | sort -k3 > "$1"
wc -l < "$1"
