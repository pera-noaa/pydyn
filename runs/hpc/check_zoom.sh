#!/bin/bash
# Read regress_zoom.sh's logs back: one line per job, PASS when it exited 0 with no event
# line (the replay prints an event only when something in it is not bitwise).
OUT=${1:?the regress directory}
fail=0
for f in "$OUT"/*.out; do
    bad=$(grep -c '^e0' "$f"); code=$(sed -n 's/^EXIT=//p' "$f")
    if [ -z "$code" ]; then s=RUNNING; fail=1
    elif [ "$code" = 0 ] && [ "$bad" = 0 ]; then s=PASS
    else s=FAIL; fail=1; fi
    echo "$s $(basename "$f") exit=${code:-none} non-bitwise-events=$bad $(sed -n 2p "$f")"
done
exit $fail
