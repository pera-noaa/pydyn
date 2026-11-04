"""`python -m pydyn.replay_zoom` is `python -m pydyn.replay`, kept for the command lines on
record (runs/MANIFEST.md): the zoom replay became the one replay, a single region a tree of one."""

import sys

from .replay import main

if __name__ == "__main__":
    sys.exit(main())
