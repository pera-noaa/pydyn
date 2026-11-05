"""Where pydyn keeps what it compiles at run time, and how it loads it back.

One directory, `${XDG_CACHE_HOME:-~/.cache}/pydyn`, created 0700 and used only while it is the
user's own and writable by nobody else: a library loaded from it runs with the user's rights,
so a shared or group-writable location (a checkout on an HPC file system, the package's own
__pycache__) is never used.  Each file's name carries a hash of what built it (the source, the
flags, the platform and the compiler's --version), so a new compiler builds anew.  The
directory is never pruned; deleting it is always safe (`rm -r ~/.cache/pydyn`).
"""

import ctypes
import hashlib
import os
import shlex
import stat
import subprocess
from functools import cache
from pathlib import Path


def cache_dir():
    """The cache directory, made if absent; RuntimeError if it is not private to this user."""
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "pydyn"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_private(root)
    return root


def _check_private(path):
    st = path.stat()
    if st.st_uid != os.geteuid() or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(
            f"pydyn: {path} is not private to you (owner uid {st.st_uid}, mode "
            f"{stat.filemode(st.st_mode)}); `chmod go-w` it, or set XDG_CACHE_HOME elsewhere"
        )


def command(env, default):
    """$env as an argument list (`CC="ccache gcc"` is two words), else [default]."""
    return shlex.split(os.environ.get(env) or default)


@cache
def compiler_id(cmd):
    """`cmd` (a string or an argument tuple) and the first line of its --version."""
    argv = list(cmd) if isinstance(cmd, (tuple, list)) else [cmd]
    try:
        out = subprocess.run([*argv, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return " ".join(argv)
    return f"{' '.join(argv)} {out.splitlines()[0] if out else ''}"


def key(*parts):
    """A short hash of everything that went into a build."""
    return hashlib.sha1("\0".join(map(str, parts)).encode()).hexdigest()[:12]


def atomic(path, make):
    """Write `path` from `make(tmpdir)`, which returns the file it built there: concurrent
    builders (a replay's worker processes) each build privately and the last rename wins.  A
    failed command becomes a RuntimeError carrying its stderr."""
    import tempfile

    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        try:
            out = make(Path(tmp))
        except subprocess.CalledProcessError as e:
            cmd = " ".join(map(str, e.cmd))
            raise RuntimeError(f"pydyn: {cmd} failed:\n{(e.stderr or '').strip()[-4000:]}") from e
        os.replace(out, path)


def built(name, build):
    """cache_dir()/name, made by `build(path)` (which writes path) if absent."""
    path = cache_dir() / name
    if not path.exists():
        build(path)
    _check_private(path)
    return path


def load_library(name, build):
    """ctypes.CDLL of cache_dir()/name, built by `build(path)` when absent, and built again
    once when the cached file will not load (a truncated or foreign file)."""
    path = built(name, build)
    try:
        return ctypes.CDLL(str(path))
    except OSError:
        path.unlink(missing_ok=True)
        return ctypes.CDLL(str(built(name, build)))
