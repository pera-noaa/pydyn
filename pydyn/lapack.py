"""The `with_lapack` matrix inverse of convection.F90 `fastminv`, called through the same
LAPACK library the model was linked with: one DGETRF + DGETRI(lwork = n) per column.

numpy.linalg.inv (LU solve against the identity, numpy's bundled LAPACK) reproduces it to a
few ulp only; calling the model's own library through ctypes reproduces it bitwise.
`lapack_library` looks in `$PYDYN_LAPACK`, then Accelerate (macOS), then `find_library`.
"""

import ctypes
import ctypes.util
import os

import numpy as np

__all__ = ["LapackInverse", "batched_inverse", "lapack_library"]

LAPACK_CANDIDATES = (
    "/System/Library/Frameworks/Accelerate.framework/Accelerate",
    "/usr/lib64/liblapack.so",
    "/usr/lib/x86_64-linux-gnu/liblapack.so",
)


def lapack_library():
    """Path of a LAPACK library exporting dgetrf_/dgetri_; None if none is found."""
    env = os.environ.get("PYDYN_LAPACK")
    if env:
        return env
    for path in (*LAPACK_CANDIDATES, ctypes.util.find_library("lapack")):
        if path and _loadable(path):  # macOS frameworks live in the dyld cache, not on disk
            return path
    return None


def _loadable(path):
    try:
        ctypes.CDLL(path)
    except OSError:
        return False
    return True


class LapackInverse:
    """Callable batched inverse, `inv = LapackInverse(lib)(a)` for a of shape (ncol, n, n)."""

    def __init__(self, lib):
        self.lib = ctypes.CDLL(lib)
        self.path = lib

    def __call__(self, a):
        ncol, n, _ = a.shape
        cn, info = ctypes.byref(ctypes.c_int(n)), ctypes.c_int(0)
        ipiv_a, work_a = np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float64)  # kept alive
        ipiv, work = _ptr(ipiv_a), _ptr(work_a)
        out = np.empty_like(a)
        for c in range(ncol):
            af = np.asfortranarray(a[c])  # DGETRF/DGETRI work in place on a column-major copy
            p = _ptr(af)
            self.lib.dgetrf_(cn, cn, p, cn, ipiv, ctypes.byref(info))
            if info.value:
                raise ValueError(f"dgetrf: info = {info.value} for column {c}")
            self.lib.dgetri_(cn, p, cn, ipiv, work, cn, ctypes.byref(info))
            if info.value:
                raise ValueError(f"dgetri: info = {info.value} for column {c}")
            out[c] = af
        return out


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def batched_inverse(lapack="auto"):
    """The inverse to hand to `kernels.convec`: LapackInverse when a library is available
    (or `lapack` is a path), numpy.linalg.inv otherwise ("numpy" / False forces numpy, "crout"
    kernels.fastminv)."""
    if lapack in ("numpy", False):
        return np.linalg.inv
    if lapack == "crout":  # the build without with_lapack: kernels.fastminv
        return "crout"
    lib = lapack_library() if lapack in ("auto", True) else lapack
    if lib is None:
        if lapack is True:
            raise RuntimeError("batched_inverse: no LAPACK library found (set PYDYN_LAPACK)")
        return np.linalg.inv
    return LapackInverse(lib)
