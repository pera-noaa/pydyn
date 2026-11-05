"""The array idioms the dynamic core is written in.  Each one is a named Fortran statement,
evaluated in the same order, so using it instead of the spelled-out slices costs nothing
bitwise.  Faces: an array of n cells along `axis` has n+1 faces, the two outer ones with no
upwind cell; `advance` takes all n+1, `upwind`/`courant` only the n-1 interior ones.

    seqsum(q, axis)          sum(q, dim)          one accumulation in index order (np.sum is
                                                  pairwise and rounds differently)
    advance(m, f, axis)      m + f(l-1) - f(l)    mass after one application of a face flux;
                                                  NOT m - np.diff(f): that rounds f(l)-f(l-1) first.
                                                  advectx's  rm + (f(i-1) - f(i))  IS  rm - np.diff(f)
    upwind(f, q, axis)       q(iup)               q in the upwind cell of each interior face
    courant(m, f, axis)      f / m(iup)           Courant number of every interior face;
                             f * m_inv(iup)       reciprocal=True is the trunk dynamu spelling
    convergence(pu, pv)      pu(i-1) - pu(i) + pv(j) - pv(j+1)   dynam0 / BalanceMassFluxes
    clusters(row, n, c)      row(1:n*c) as (n, c, ...)  the reduced-grid cells of a latitude row
    wrap(q, axis)            q(0) = q(n), q(n+1) = q(1)          the periodic halo of a row
    closed(q, axis)          q(0) = q(n+1) = 0                   zero outer faces
    subcycles(m, f, ...)     the CFL sub-cycle search of every advection operator
    libm.exp/log/pow/cos/sin the C library's functions on every value, as the compiled Fortran
                             calls them -- a C loop through each function's own address,
                             compiled on first use (else np.vectorize, one Python call per
                             value; PYDYN_LIBM_SCALAR=1 forces that); NumPy's vectorised ones
                             may round differently.
                             Out of the domain (log of a negative, a negative base to a
                             fractional power) they give NaN / -inf as the Fortran does.
                             That is the system libm, which is what gfortran calls; ifort
                             calls its own libimf, which rounds differently (2-3 ulp in the
                             chemistry's rates).  $PYDYN_LIBM=<path of libimf.so> -- or
                             `use_libm(path)` -- routes all five through that library instead.
    libm.cube(x)             x**3: gfortran expands it to (x*x)*x; ifort calls libimf's
                             __powr8i4, 1 ulp off that on a quarter of values, which
                             use_libm picks up when the library has it.
"""

import ctypes
import math
import os

import numpy as np

# dims.F90 `one`: the threshold every CFL comparison in the model is written against
# (`abs(alpha) >= one`).  Named rather than spelled 1.0 so tests/test_constants.py can pin
# it to the Fortran -- a model that moved it off 1.0 would change which steps are legal.
ONE = 1.0

__all__ = [
    "ONE",
    "advance",
    "closed",
    "clusters",
    "convergence",
    "courant",
    "fresh",
    "libm",
    "nint",
    "progressive",
    "seqsum",
    "subcycles",
    "upwind",
    "use_libm",
    "wrap",
]


def _log(x):
    if x > 0.0:
        return math.log(x)
    return -math.inf if x == 0.0 else math.nan


def _pow(x, y):
    try:
        return math.pow(x, y)
    except ValueError:  # negative base, fractional exponent
        return math.nan


def nint(x):
    """Fortran nint: round half away from zero (np.rint and round() are half-to-even)."""
    return math.floor(abs(x) + 0.5) * (1 if x >= 0 else -1)


def _cube(x):
    return (x * x) * x


# A C loop that applies a library function to every element, through the function's own
# address: the same call the scalar path makes, without one Python call per value.  Compiled on
# the first libm call, not at import; without a C compiler the scalar np.vectorize path is kept
# (with a warning).  The loop only calls f, so the compiler cannot change a value; it is in the
# cache key anyway (compiler_id).
_MAP_SOURCE = r"""
#include <stddef.h>
void map1(double (*f)(double), const double *x, double *y, size_t n)
{ for (size_t i = 0; i < n; i++) y[i] = f(x[i]); }
void map2(double (*f)(double, double), const double *x, const double *z, double *y, size_t n)
{ for (size_t i = 0; i < n; i++) y[i] = f(x[i], z[i]); }
void mapi(double (*f)(double, int), const double *x, int k, double *y, size_t n)
{ for (size_t i = 0; i < n; i++) y[i] = f(x[i], k); }
"""


def compiler_id(cmd):
    """`cmd` and the first line of its --version: what a cached build's key must carry."""
    import subprocess

    try:
        out = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return cmd
    return f"{cmd} {out.splitlines()[0] if out else ''}"


def _map_library():
    import hashlib
    import platform
    import subprocess
    import tempfile
    import warnings
    from pathlib import Path

    cc = os.environ.get("CC", "cc")
    tag = hashlib.sha1((_MAP_SOURCE + platform.platform() + compiler_id(cc)).encode()).hexdigest()[:12]
    errors = []
    for root in (Path(__file__).parent / "__pycache__", Path.home() / ".cache" / "pydyn"):
        so = root / f"_libm_map_{tag}.so"
        if so.exists():
            return ctypes.CDLL(str(so))
        try:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=root) as tmp:
                src, out = Path(tmp) / "map.c", Path(tmp) / "map.so"
                src.write_text(_MAP_SOURCE)
                subprocess.run([cc, "-O1", "-shared", "-fPIC", "-o", str(out), str(src)], check=True,
                               capture_output=True, text=True)  # fmt: skip
                os.replace(out, so)  # atomic: concurrent workers may build it at once
            return ctypes.CDLL(str(so))
        except subprocess.CalledProcessError as e:
            errors.append(f"{root}: {e.stderr.strip()[:500]}")
        except OSError as e:
            errors.append(f"{root}: {e}")
    warnings.warn(f"pydyn.ops: no C loop for libm ({cc}), one value at a time: " + "; ".join(errors),
                  stacklevel=3)
    return None


_MAP_BUILT = []


def _map():
    """The compiled loop, built on first use (None: scalar; PYDYN_LIBM_SCALAR=1 forces that)."""
    if not _MAP_BUILT:
        _MAP_BUILT.append(_map_library() if os.environ.get("PYDYN_LIBM_SCALAR") != "1" else None)
    return _MAP_BUILT[0]


_DP = ctypes.POINTER(ctypes.c_double)


def _mapped(fn, nargs, k=None):
    """fn (a ctypes function of the library) over arrays, through the C loop."""
    addr = ctypes.cast(fn, ctypes.c_void_p)

    loop = _map()

    def call(*args):
        arrays = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in args))
        shape = arrays[0].shape
        xs = [np.ascontiguousarray(a).ravel() for a in arrays]
        y = np.empty(xs[0].size)
        ptrs = [x.ctypes.data_as(_DP) for x in xs]
        if k is not None:
            loop.mapi(addr, ptrs[0], ctypes.c_int(k), y.ctypes.data_as(_DP), ctypes.c_size_t(y.size))
        elif nargs == 1:
            loop.map1(addr, ptrs[0], y.ctypes.data_as(_DP), ctypes.c_size_t(y.size))
        else:
            loop.map2(addr, ptrs[0], ptrs[1], y.ctypes.data_as(_DP), ctypes.c_size_t(y.size))
        return y.reshape(shape)

    return call


def _bind(lib):
    """libm.exp/log/cos/sin/pow from `lib`'s functions, each over whole arrays when the C loop
    is built, else one value at a time (np.vectorize)."""
    for name, nargs in (("exp", 1), ("log", 1), ("cos", 1), ("sin", 1), ("pow", 2)):
        f = getattr(lib, name)
        f.restype, f.argtypes = ctypes.c_double, [ctypes.c_double] * nargs
        setattr(libm, name, staticmethod(_mapped(f, nargs) if _map() else np.vectorize(f, otypes=[float])))


_SCALAR = {  # the C library's functions one value at a time: math.* calls them
    "exp": np.vectorize(math.exp, otypes=[float]),
    "log": np.vectorize(_log, otypes=[float]),
    "pow": np.vectorize(_pow, otypes=[float]),
    "cos": np.vectorize(math.cos, otypes=[float]),
    "sin": np.vectorize(math.sin, otypes=[float]),
}


def _first_call(name):
    """libm.<name> until the first call, which binds all five (the system libm through the C
    loop, else _SCALAR) and makes it."""

    def call(*args):
        for key, f in _SCALAR.items():
            if getattr(libm.__dict__[key], "__func__", None) is _STUBS[key]:
                setattr(libm, key, staticmethod(f))
        _system_libm()
        return getattr(libm, name)(*args)

    return call


_STUBS = {name: _first_call(name) for name in _SCALAR}


class libm:  # a namespace, not a class to instantiate
    cube = staticmethod(_cube)
    exp = staticmethod(_STUBS["exp"])
    log = staticmethod(_STUBS["log"])
    pow = staticmethod(_STUBS["pow"])
    cos = staticmethod(_STUBS["cos"])
    sin = staticmethod(_STUBS["sin"])
    path = None  # the library, when not the system libm


def use_libm(path):
    """Route libm.* through the C library at `path` (ifort's libimf.so for an Intel build)."""
    lib = ctypes.CDLL(path)
    _bind(lib)
    if hasattr(lib, "__powr8i4"):  # ifort's integer power, x**3
        powi = lib.__powr8i4
        powi.restype, powi.argtypes = ctypes.c_double, [ctypes.c_double, ctypes.c_int]
        cube = _mapped(powi, 1, k=3) if _map() else np.vectorize(lambda x: powi(x, 3), otypes=[float])
        libm.cube = staticmethod(cube)
    libm.path = path


def _system_libm():
    """The C library's own exp/log/cos/sin/pow -- what math.* calls -- over whole arrays."""
    import ctypes.util

    name = ctypes.util.find_library("m") or ctypes.util.find_library("c")
    if name and _map():
        _bind(ctypes.CDLL(name))


if os.environ.get("PYDYN_LIBM"):
    use_libm(os.environ["PYDYN_LIBM"])


def seqsum(q, axis=0):
    q = np.moveaxis(q, axis, 0)
    acc = q[0].copy()
    for term in q[1:]:
        acc = acc + term
    return acc


def _faces(f, axis):
    """(lower, upper) face views of a flux array along `axis`."""
    lo = [slice(None)] * f.ndim
    hi = list(lo)
    lo[axis], hi[axis] = slice(None, -1), slice(1, None)
    return f[tuple(lo)], f[tuple(hi)]


def _interior(f, axis):
    ix = [slice(None)] * f.ndim
    ix[axis] = slice(1, -1)
    return f[tuple(ix)]


def advance(m, f, axis=0):
    lo, hi = _faces(f, axis)
    return m + lo - hi


def upwind(f, q, axis=0):
    lo, hi = _faces(q, axis)  # the cells below and above each interior face
    return np.where(f >= 0.0, lo, hi)


def courant(m, f, axis=0, reciprocal=False):
    if reciprocal:
        return f * upwind(f, 1.0 / m, axis)
    return f / upwind(f, m, axis)


def convergence(pu, pv):
    return pu[:-1] - pu[1:] + pv[:, :-1] - pv[:, 1:]


def clusters(row, n, c):
    return row[: n * c].reshape(n, c, *row.shape[1:])


def _halo(q, axis, periodic):
    """q with one halo cell at each end of `axis` (np.pad's wrap / constant, without its overhead)."""
    q = np.asarray(q)
    axis %= q.ndim
    n, pre = q.shape[axis], (slice(None),) * axis
    out = np.empty((*q.shape[:axis], n + 2, *q.shape[axis + 1 :]), dtype=q.dtype)
    out[(*pre, slice(1, n + 1))] = q
    if periodic:
        out[(*pre, 0)], out[(*pre, n + 1)] = q[(*pre, n - 1)], q[(*pre, 0)]
    else:
        out[(*pre, 0)] = out[(*pre, n + 1)] = 0
    return out


def wrap(q, axis=0):
    return _halo(q, axis, True)


def closed(q, axis=0):
    return _halo(q, axis, False)


# ------------------------------------------------------------------ the CFL sub-cycle search
def progressive(f, n, f0):
    """dynamu, dynamv: the current flux scaled again, in place -- am*nloop/(nloop+1), cumulatively."""
    f[...] = f * (n - 1) / n
    return f


def fresh(f, n, f0):
    """dynamw: the original flux divided -- cm/n_advz.  Rounds differently from `progressive`."""
    return f0 / float(n)


def subcycles(m, f, nmax, shrink, axis=0, periodic=False, seen=None):
    """The smallest nloop for which nloop trial sub-steps of the mass field alone stay under
    CFL (|Courant| < 1 on every interior face), shrinking the flux between attempts with
    `shrink(f, n, f0)` (`progressive` or `fresh`).  A periodic axis wraps, so every face has an
    upwind cell.  `seen(courant, ok)` is told the |Courant numbers| of every trial sub-step.
    Returns (nloop, flux to use); None once nloop would exceed nmax."""
    f0, n = f, 1
    while not _trial(m, f, n, axis, periodic, seen):
        n += 1
        f = shrink(f, n, f0)
        if n > nmax:
            return None
    return n, f


def _trial(m, f, n, axis, periodic, seen):
    mx = m.copy()
    for _ in range(n):
        cells, faces = (wrap(mx, axis), f) if periodic else (mx, _interior(f, axis))
        c = np.abs(courant(cells, faces, axis))
        ok = bool(np.all(c < ONE))
        if seen is not None:
            seen(c, ok)
        if not ok:
            return False
        mx = advance(mx, f, axis)
    return True
