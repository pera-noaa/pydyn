"""The emulated device: arrays that live in host memory but behave like CuPy's, so a run on the
emulation backend is also a residency check of the device code.

`EmuArray` wraps a NumPy array and refuses what CuPy refuses: an operation mixing it with a
host array (anything with ndim > 0), `np.asarray` of it, an assignment of a host array into it.
NumPy functions called on it dispatch through `__array_function__` / `__array_ufunc__` to the
same NumPy function on the wrapped data, as CuPy's do to CuPy's -- so the arithmetic is NumPy's
own, and a transfer only happens through `Backend.to_device` / `to_host`, which count it.
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

__all__ = ["EmuArray", "ResidencyError", "emu_namespace"]


class ResidencyError(TypeError):
    """A host array met a device array without an explicit transfer."""


def _is_host_array(x):
    return isinstance(x, np.ndarray) and x.ndim > 0


def _unwrap(x):
    if isinstance(x, EmuArray):
        return x._a
    if _is_host_array(x):
        raise ResidencyError(f"implicit host -> device transfer of a {x.shape} array")
    if isinstance(x, (list, tuple)):
        return type(x)(_unwrap(v) for v in x)
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    return x


def _wrap(x):
    if isinstance(x, np.ndarray):
        return EmuArray(x)
    if isinstance(x, tuple):
        return tuple(_wrap(v) for v in x)
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _key(key):
    """Indices may be device arrays (CuPy's own) or plain integers / slices."""
    if isinstance(key, tuple):
        return tuple(_key(k) for k in key)
    if isinstance(key, EmuArray):
        return key._a
    if _is_host_array(key):
        raise ResidencyError("indexing a device array with a host index array")
    return key


class EmuArray(NDArrayOperatorsMixin):
    __slots__ = ("_a",)

    def __init__(self, a):
        self._a = a

    # -- what a CuPy array refuses
    def __array__(self, dtype=None, copy=None):
        raise ResidencyError("implicit device -> host conversion (np.asarray of a device array)")

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        if out is not None:
            kwargs["out"] = tuple(_unwrap(o) for o in out)
        return _wrap(getattr(ufunc, method)(*_unwrap(inputs), **_unwrap(kwargs)))

    def __array_function__(self, func, types, args, kwargs):
        return _wrap(func(*_unwrap(args), **_unwrap(kwargs)))

    # -- the array
    def __getitem__(self, key):
        return _wrap(self._a[_key(key)])

    def __setitem__(self, key, value):
        self._a[_key(key)] = _unwrap(value)

    def __len__(self):
        return len(self._a)

    def __bool__(self):
        return bool(self._a)

    def __float__(self):
        return float(self._a)

    def __int__(self):
        return int(self._a)

    def __index__(self):
        return self._a.__index__()

    def __repr__(self):
        return f"EmuArray({self._a!r})"

    def __getattr__(self, name):
        if name.startswith("__"):  # __array_interface__ and kin would let NumPy read the buffer
            raise AttributeError(name)
        attr = getattr(self._a, name)
        if callable(attr):
            return lambda *a, **k: _wrap(attr(*_unwrap(a), **_unwrap(k)))
        return _wrap(attr) if isinstance(attr, np.ndarray) else attr

    @property
    def T(self):
        return EmuArray(self._a.T)

    # the launcher's and boxes' reads, without __getattr__
    dtype = property(lambda self: self._a.dtype)
    shape = property(lambda self: self._a.shape)
    strides = property(lambda self: self._a.strides)
    ndim = property(lambda self: self._a.ndim)
    size = property(lambda self: self._a.size)
    itemsize = property(lambda self: self._a.itemsize)

    def item(self):
        return self._a.item()

    def get(self):  # cupy.ndarray.get: an explicit transfer; Backend.to_host is the counted one
        raise ResidencyError("use Backend.to_host, which counts the transfer")

    @property
    def base_array(self):
        """The wrapped host buffer, for the kernel launcher only."""
        return self._a


_CREATE = ("zeros", "empty", "ones", "full")  # whose default dtype is the backend's real


class _Namespace:
    """`xp` for the emulation backend: numpy's functions, device arrays in and out."""

    def __init__(self, real=np.float64):
        self.real = real

    def __getattr__(self, name):
        f = getattr(np, name)
        if not callable(f):
            return f
        if isinstance(f, type):  # dtypes
            return f
        real = self.real

        def call(*args, **kwargs):
            if name in _CREATE and "dtype" not in kwargs and len(args) < (3 if name == "full" else 2):
                kwargs["dtype"] = real
            return _wrap(f(*_unwrap(args), **_unwrap(kwargs)))

        return call

    def asarray(self, x, dtype=None):
        if isinstance(x, EmuArray):
            return x if dtype is None else EmuArray(x._a.astype(dtype, copy=False))
        if _is_host_array(x):
            raise ResidencyError("xp.asarray of a host array: use Backend.to_device")
        return EmuArray(np.asarray(x, dtype=dtype))


_NAMESPACES = {}


def emu_namespace(real=np.float64):
    real = np.dtype(real).type
    if real not in _NAMESPACES:
        _NAMESPACES[real] = _Namespace(real)
    return _NAMESPACES[real]
