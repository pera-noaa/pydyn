#!/usr/bin/env bash
# Lay this repo's HEAD over a clone of the public repo (pera-noaa/pydyn) as one new commit.
#
#   runs/export_public.sh PUBLIC_CLONE              stage it and show what would change
#   runs/export_public.sh PUBLIC_CLONE -m MESSAGE   and commit it (never pushes)
#
# The public tree is HEAD's tracked files without CLAUDE.md, so files deleted here are deleted
# there.  Refuses on: uncommitted changes here; a public clone that is not clean, not on main,
# or behind origin/main; anything in the result that cites CLAUDE.md; a /Users/ path outside
# runs/MANIFEST.md (whose provenance records name the laptop's); a token-shaped string.  This
# script and its test are exempt from the three checks, since they spell out what they look for.
set -euo pipefail
die() { echo "export_public: $*" >&2; exit 1; }

case $# in
    1) MSG= ;;
    3) [ "$2" = -m ] || die "usage: $0 PUBLIC_CLONE [-m MESSAGE]"; MSG=$3 ;;
    *) die "usage: $0 PUBLIC_CLONE [-m MESSAGE]" ;;
esac
LAB=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PUB=$(cd "$1" && pwd)

[ -z "$(git -C "$LAB" status --porcelain --untracked-files=no)" ] || die "uncommitted changes in $LAB"
[ "$(git -C "$PUB" rev-parse --show-toplevel)" = "$PUB" ] || die "$PUB is not a git clone's top"
[ "$(git -C "$PUB" branch --show-current)" = main ] || die "$PUB is not on main"
[ -z "$(git -C "$PUB" status --porcelain)" ] || die "$PUB is not clean"
git -C "$PUB" fetch -q origin main
[ -z "$(git -C "$PUB" rev-list HEAD..origin/main)" ] || die "$PUB is behind origin/main: pull first"

git -C "$PUB" rm -rq --ignore-unmatch .
git -C "$LAB" archive HEAD | tar -x -C "$PUB"
rm -f "$PUB/CLAUDE.md"
git -C "$PUB" add -A

# This script and its test name the strings the first three checks look for, so those skip
# them; the token check skips nothing (the test builds its token at run time).
SELF=(':!runs/export_public.sh' ':!tests/test_export_public.py')
bad=0
if hits=$(git -C "$PUB" grep -n --cached 'CLAUDE\.md' -- "${SELF[@]}"); then
    echo "$hits" >&2; echo "export_public: the public tree cites CLAUDE.md" >&2; bad=1; fi
if hits=$(git -C "$PUB" grep -n --cached '/Users/' -- ':!runs/MANIFEST.md' "${SELF[@]}"); then
    echo "$hits" >&2; echo "export_public: a laptop path outside runs/MANIFEST.md" >&2; bad=1; fi
if hits=$(git -C "$PUB" grep -nE --cached '/(projects|scratch/alpine|home|glade/u/home)/[a-z]{4}[0-9]{4}' -- "${SELF[@]}"); then
    echo "$hits" >&2; echo "export_public: a cluster username in a path (write \$USER)" >&2; bad=1; fi
if hits=$(git -C "$PUB" grep -nE --cached 'ghp_[A-Za-z0-9]{20}|github_pat_[A-Za-z0-9_]{20}|xox[bp]-[0-9]|BEGIN [A-Z ]*PRIVATE KEY'); then
    echo "$hits" >&2; echo "export_public: a token-shaped string" >&2; bad=1; fi
if [ $bad -ne 0 ]; then
    git -C "$PUB" reset -q --hard HEAD
    die "refused; $PUB is back at its HEAD"
fi

if git -C "$PUB" diff --cached --quiet; then
    echo "export_public: nothing to export, $PUB already matches $(git -C "$LAB" rev-parse --short HEAD)"
    exit 0
fi
git -C "$PUB" diff --cached --stat
if [ -z "$MSG" ]; then
    echo "export_public: staged in $PUB from lab $(git -C "$LAB" rev-parse --short HEAD); not committed (pass -m)"
    exit 0
fi
git -C "$PUB" -c user.name="Ash Pera" -c user.email="ashley.pera@noaa.gov" commit -q -m "$MSG"
echo "export_public: committed $(git -C "$PUB" rev-parse --short HEAD); push with: git -C $PUB push origin main"
