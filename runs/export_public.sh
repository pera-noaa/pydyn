#!/usr/bin/env bash
# Lay this repo's HEAD over a clone of the public repo (pera-noaa/pydyn) as one new commit.
#
#   runs/export_public.sh PUBLIC_CLONE              stage it and show what would change
#   runs/export_public.sh PUBLIC_CLONE -m MESSAGE   and commit it (never pushes)
#
# The public tree is HEAD's tracked files without any CLAUDE.md / CLAUDE.local.md, so files
# deleted here are deleted there.  Refuses on: a PUBLIC_CLONE that is empty, missing, this repo,
# not a clone's top, not on main, not clean, behind origin/main, or whose origin is not
# $PYDYN_PUBLIC_REMOTE (default pera-noaa/pydyn); uncommitted changes here; and anything
# runs/leak_scan.py finds in the result (CLAUDE citations, laptop paths, cluster usernames,
# token-shaped strings, unlisted emails and hosts).  On a refusal after staging, the clone is
# reset to its HEAD.
set -euo pipefail
die() { echo "export_public: $*" >&2; exit 1; }

case $# in
    1) MSG= ;;
    3) [ "$2" = -m ] || die "usage: $0 PUBLIC_CLONE [-m MESSAGE]"; MSG=$3 ;;
    *) die "usage: $0 PUBLIC_CLONE [-m MESSAGE]" ;;
esac
[ -n "$1" ] || die "PUBLIC_CLONE is empty"
[ -d "$1" ] || die "$1 is not a directory"
LAB=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
PUB=$(cd "$1" && pwd -P)
REMOTE=${PYDYN_PUBLIC_REMOTE:-pera-noaa/pydyn}

[ "$PUB" != "$(cd "$LAB" && pwd -P)" ] || die "PUBLIC_CLONE is this repo"
[ -z "$(git -C "$LAB" status --porcelain --untracked-files=no)" ] || die "uncommitted changes in $LAB"
[ "$(git -C "$PUB" rev-parse --show-toplevel 2>/dev/null)" = "$PUB" ] || die "$PUB is not a git clone's top"
url=$(git -C "$PUB" remote get-url origin 2>/dev/null) || die "$PUB has no origin"
case $url in *"$REMOTE"|*"$REMOTE.git"|*"$REMOTE/") ;; *) die "$PUB's origin is $url, not $REMOTE" ;; esac
[ "$(git -C "$PUB" branch --show-current)" = main ] || die "$PUB is not on main"
[ -z "$(git -C "$PUB" status --porcelain)" ] || die "$PUB is not clean"
git -C "$PUB" fetch -q origin main
[ -z "$(git -C "$PUB" rev-list HEAD..origin/main)" ] || die "$PUB is behind origin/main: pull first"

git -C "$PUB" rm -rq --ignore-unmatch .
git -C "$LAB" archive HEAD | tar -x -C "$PUB"
git -C "$LAB" ls-files | { grep -E '(^|/)CLAUDE(\.local)?\.md$' || true; } | while IFS= read -r f; do
    rm -f "$PUB/$f"
done
git -C "$PUB" add -A

if ! python3 "$LAB/runs/leak_scan.py" "$PUB"; then
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
