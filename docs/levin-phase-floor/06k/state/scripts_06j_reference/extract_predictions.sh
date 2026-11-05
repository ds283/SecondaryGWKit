#!/bin/bash
# levin-phase-floor 06a section 3.2: copy the prompt's section 4 byte for byte, from its heading
# "## 4. Predictions" up to (not including) the line "## 5. Descriptive, not scored". New in 06a.
# Usage: extract_predictions.sh PROMPT OUT
set -eu
awk '/^## 4\. Predictions$/{p=1} /^## 5\. Descriptive, not scored$/{p=0} p' "$1" > "$2"
