"""The Metal backend: pydyn.gpu's kernels on an Apple GPU, in float32 (the GPU has no double).

csrc/ is compiled offline by `xcrun metal -fno-fast-math -ffp-contract=off` (the runtime
compiler cannot turn contraction off, and contracts a*b+c by default) into a .metallib;
csrc/metal_shim.mm is the runtime, called through ctypes.

Memory is one shared-storage arena that the GPU addresses by gpuAddress and the CPU as NumPy
views -- on Apple silicon they are the same memory, so "on the device" means in the arena, and
an upload or download is a copy within it, counted by the ledger as on any backend.  A kernel's
arguments are one struct (`<kernel>_args`, generated from its C signature) of device pointers
and scalars; dispatches queue on one compute encoder, in order, with a barrier between them.

`MetalArray` is EmuArray's residency discipline (no implicit host <-> device mixing) on arena
memory.  The CPU may compute on it -- the driver's small exchange and bookkeeping operations
do -- but only after the queued kernels have run: any touch of the data but a view syncs
first, and memory freed while kernels are queued is only reused after the next sync.  The
counters `cpu_ops` and `syncs` say how often that happened.
"""

import ctypes
import hashlib
import os
import re
import struct
import subprocess
import tempfile
import time
import weakref
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import numpy as np

from . import convec_gen, emu
from . import coop as lines
from .backend import CSRC, ROUTED, SOURCES, Backend
from .emu import EmuArray, ResidencyError

__all__ = ["MetalArray", "MetalBackend", "metal_source"]

_ALIGN = 256
_ARENA = int(os.environ.get("PYDYN_METAL_ARENA_MB", "4096")) << 20
_CONTRACT = False  # True: compile with -ffast-math, a*b+c contracted (tests: the gate can fail)
_COOPERATIVE_SOURCES = ("convec_metal.h",)  # Metal-only kernels that take a csrc kernel's place
_KERNEL_SOURCES = [n for n in SOURCES if n.endswith((".h", ".cuh")) and n not in _COOPERATIVE_SOURCES]
_CONVEC_CPT = 2  # convec_metal.h's columns a threadgroup (PD_CONVEC_CPT)
_FIELD = {"i64": ("long", "q", 8), "int": ("int", "i", 4), "real": ("real", "f", 4)}


def _cache_dir():
    for root in (Path(__file__).parent / "__pycache__", Path.home() / ".cache" / "pydyn"):
        try:
            root.mkdir(parents=True, exist_ok=True)
            return root
        except OSError:
            continue
    raise RuntimeError("pydyn.gpu.metal: no cache directory")


# ------------------------------------------------------------------ the source
def metal_source(signatures, defines=None):
    """csrc/'s kernels as one MSL translation unit: PD_F32, the includes inlined, every
    PD_GLOBAL kernel turned into a body taking (pd_tid, pd_nth) and a `kernel` reading its
    arguments from a `<name>_args` struct."""
    text = "".join(
        re.sub(r'^#include "[^"]+"\n', "", (CSRC / n).read_text(), flags=re.M) for n in _KERNEL_SOURCES
    )
    text = re.sub(
        r"PD_GLOBAL\s+void\s+(\w+)\s*\(", r"PD_GLOBAL void \1__body(uint pd_tid, uint pd_nth, ", text
    )
    head = ["#define PD_F32 1"] + [f"#define {k} {v}" for k, v in sorted((defines or {}).items())]
    wrappers = []
    for name, sig in signatures.items():
        fields = []
        for ctype, pname in sig:
            if ctype.endswith("*"):
                fields.append(f"    device {ctype[:-1].strip()} *{pname};")
            else:
                fields.append(f"    {_FIELD[ctype][0]} {pname};")
        call = ", ".join(f"a.{p}" for _, p in sig)
        params = "uint pd_tid [[thread_position_in_grid]], uint pd_nth [[threads_per_grid]]"
        wrappers.append(
            f"struct {name}_args {{\n" + "\n".join(fields) + "\n};\n"
            f"kernel void {name}(constant {name}_args &a [[buffer(0)]], {params})\n"
            f"{{\n    {name}__body(pd_tid, pd_nth, {call});\n}}\n"
        )
    coop = "".join((CSRC / name).read_text() for name in _COOPERATIVE_SOURCES)
    if "PD_CONVEC_N" in (defines or {}):
        coop += convec_gen.convec_source(int(defines["PD_CONVEC_N"]))
    if "PD_COOP_FACES" in (defines or {}):  # the line kernels a SIMD group a line
        coop += lines.source(signatures, "metal")
    return "\n".join(head) + "\n" + text + "\n" + "\n".join(wrappers) + "\n" + coop


_LAYOUTS = {}  # id(signature) -> (signature, struct.Struct, pointer positions)


def _layout(sig):
    """The `<name>_args` struct of a signature: natural alignment, as MSL lays it out."""
    got = _LAYOUTS.get(id(sig))
    if got is not None and got[0] is sig:
        return got
    fmt, ptrs, off = [], [], 0
    for i, (ctype, _) in enumerate(sig):
        if ctype.endswith("*"):
            code, size = "Q", 8
            ptrs.append(i)
        else:
            _, code, size = _FIELD[ctype]
        pad = -off % size
        if pad:
            fmt.append(f"{pad}x")
        fmt.append(code)
        off += pad + size
    fmt.append(f"{-off % 8}x")
    got = _LAYOUTS[id(sig)] = (sig, struct.Struct("<" + "".join(fmt)), ptrs)
    return got


def _pack(sig, args, address):
    """The `<name>_args` struct's bytes."""
    _, st, ptrs = _layout(sig)
    vals = list(args)
    for i in ptrs:
        a = vals[i]
        vals[i] = 0 if a is None else address(a)
    return st.pack(*vals)


def _shim():
    src = CSRC / "metal_shim.mm"
    tag = hashlib.sha1(src.read_bytes()).hexdigest()[:12]
    lib = _cache_dir() / f"_pd_metal_shim_{tag}.dylib"
    if not lib.exists():
        with tempfile.TemporaryDirectory(dir=lib.parent) as tmp:
            out = Path(tmp) / "shim.dylib"
            subprocess.run(
                [
                    "clang++",
                    "-O2",
                    "-fobjc-arc",
                    "-shared",
                    "-fPIC",
                    "-framework",
                    "Metal",
                    "-framework",
                    "Foundation",
                    str(src),
                    "-o",
                    str(out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(out, lib)
    L = ctypes.CDLL(str(lib))
    L.pdm_init.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    L.pdm_library.restype = ctypes.c_void_p
    L.pdm_library.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    L.pdm_pipeline.restype = ctypes.c_void_p
    L.pdm_pipeline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    L.pdm_dispatch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_int]
    L.pdm_sync.argtypes = [ctypes.c_char_p, ctypes.c_int]
    L.pdm_mark.restype = ctypes.c_long
    L.pdm_progress.argtypes = [ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long)]
    L.pdm_wait.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_int]
    L.pdm_stats.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long),
    ]
    return L


def _metallib(source):
    tag = hashlib.sha1((source + str(_CONTRACT)).encode()).hexdigest()[:12]
    lib = _cache_dir() / f"_pd_metal_{tag}.metallib"
    if not lib.exists():
        with tempfile.TemporaryDirectory(dir=lib.parent) as tmp:
            src, air, out = Path(tmp) / "pd.metal", Path(tmp) / "pd.air", Path(tmp) / "pd.metallib"
            src.write_text(source)
            flags = ["-ffast-math"] if _CONTRACT else ["-fno-fast-math", "-ffp-contract=off"]
            r = subprocess.run(
                ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.0", *flags, "-c", str(src), "-o", str(air)],
                capture_output=True,
                text=True,
            )
            if r.returncode:
                raise RuntimeError(f"pydyn.gpu.metal: the Metal compiler failed:\n{r.stderr[-6000:]}")
            subprocess.run(
                ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(out)],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(out, lib)
    return lib


# ------------------------------------------------------------------ the arena
class _Arena:
    """First-fit allocator over the shared buffer; frees wait for the next sync."""

    def __init__(self, base, gpu, size, queued=lambda: 0):
        self.base, self.gpu, self.size = base, gpu, size
        self.free = [(0, size)]  # sorted, coalesced
        self.pending = []  # (offset, bytes, the queue's tag when freed: _Runtime.queued)
        self.queued = queued
        self.used = self.peak = 0

    def alloc(self, nbytes):
        n = max(_ALIGN, -(-nbytes // _ALIGN) * _ALIGN)
        for k, (off, size) in enumerate(self.free):
            if size >= n:
                self.free[k : k + 1] = [(off + n, size - n)] if size > n else []
                self.used += n
                self.peak = max(self.peak, self.used)
                return off, n
        raise MemoryError(f"pydyn.gpu.metal: arena of {self.size >> 20} MB full (PYDYN_METAL_ARENA_MB)")

    def release(self, off, n):
        self.pending.append((off, n, self.queued()))

    def collect(self, retired=None):
        """What was freed before the queue's progress `retired` (all of it: at a sync): no
        kernel still queued can reference it."""
        done = [(o, n) for o, n, q in self.pending if retired is None or q < retired]
        if not done:
            return
        self.pending = [b for b in self.pending if not (retired is None or b[2] < retired)]
        blocks = sorted(self.free + done)
        merged = []
        for off, n in blocks:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((off, n))
        self.free = merged
        self.used = self.size - sum(n for _, n in merged)

    def array(self, shape, dtype):
        dtype = np.dtype(dtype)
        shape = (shape,) if np.isscalar(shape) else tuple(int(s) for s in shape)
        count = int(np.prod(shape, dtype=np.int64))
        off, n = self.alloc(count * dtype.itemsize)
        # one ctypes block per allocation: every view's base chain ends at it (NumPy collapses
        # chains of ndarrays, not this), so it dies with the last view
        block = (ctypes.c_uint8 * max(count * dtype.itemsize, dtype.itemsize)).from_address(self.base + off)
        weakref.finalize(block, self.release, off, n)
        return np.frombuffer(block, dtype=dtype, count=count).reshape(shape)

    def contains(self, a):
        p = a.__array_interface__["data"][0]
        return self.base <= p < self.base + self.size


# ------------------------------------------------------------------ the array
_VIEW_ATTRS = {
    "shape",
    "dtype",
    "strides",
    "ndim",
    "size",
    "itemsize",
    "nbytes",
    "reshape",
    "transpose",
    "view",
    "swapaxes",
    "squeeze",
    "flags",
}


def _basic(key):
    if isinstance(key, tuple):
        return all(_basic(k) for k in key)
    return key is None or key is Ellipsis or isinstance(key, (int, np.integer, slice))


class MetalArray(EmuArray):
    __slots__ = ("_gpu",)  # its GPU address, once asked (MetalBackend.address)
    be = None  # the MetalBackend, set when it is created

    def _touch(self, kind):
        self.be.cpu_touch(kind)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        self._touch(f"ufunc:{ufunc.__name__}")
        if out is not None:
            kwargs["out"] = tuple(emu._unwrap(o) for o in out)
        return self.be.adopt(getattr(ufunc, method)(*emu._unwrap(inputs), **emu._unwrap(kwargs)))

    def __array_function__(self, func, types, args, kwargs):
        self._touch(f"func:{func.__name__}")
        return self.be.adopt(func(*emu._unwrap(args), **emu._unwrap(kwargs)))

    def __getitem__(self, key):
        if _basic(key):
            return MetalArray(self._a[key])
        self._touch("getitem")
        return self.be.adopt(self._a[emu._key(key)])

    def __setitem__(self, key, value):
        self._touch("setitem")
        self._a[emu._key(key)] = emu._unwrap(value)

    def __bool__(self):
        self._touch("scalar")
        return bool(self._a)

    def __float__(self):
        self._touch("scalar")
        return float(self._a)

    def __int__(self):
        self._touch("scalar")
        return int(self._a)

    def __index__(self):
        self._touch("scalar")
        return self._a.__index__()

    def item(self):
        self._touch("scalar")
        return self._a.item()

    def copy(self):
        self._touch("copy")
        out = self.be.arena.array(self._a.shape, self._a.dtype)
        np.copyto(out, self._a)
        return MetalArray(out)

    def __repr__(self):
        return f"MetalArray(shape={self._a.shape}, dtype={self._a.dtype})"

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        attr = getattr(self._a, name)
        if name not in _VIEW_ATTRS:
            self._touch(f"attr:{name}")
        if callable(attr):
            return lambda *a, **k: self.be.adopt(attr(*emu._unwrap(a), **emu._unwrap(k)))
        return self.be.adopt(attr) if isinstance(attr, np.ndarray) else attr

    @property
    def T(self):
        return MetalArray(self._a.T)


class _Namespace:
    """`xp` for metal: allocations in the arena; anything else NumPy's, on the CPU, after a sync."""

    _CREATE: ClassVar = {"zeros": 0, "empty": None, "ones": 1}

    def __init__(self, be):
        self.be = be

    def __getattr__(self, name):
        f = getattr(np, name)
        if not callable(f) or isinstance(f, type):
            return f
        be = self.be
        if name in self._CREATE:
            fill = self._CREATE[name]

            def create(shape, dtype=None):
                a = be.arena.array(shape, be.real if dtype is None else dtype)
                if fill is not None:
                    a[...] = fill
                return MetalArray(a)

            return create

        def call(*args, **kwargs):
            be.cpu_touch(f"xp:{name}")
            return be.adopt(f(*emu._unwrap(args), **emu._unwrap(kwargs)))

        return call

    def zeros_like(self, a, dtype=None):
        return self.zeros(a.shape, a.dtype if dtype is None else dtype)

    def empty_like(self, a, dtype=None):
        return self.empty(a.shape, a.dtype if dtype is None else dtype)

    def asarray(self, x, dtype=None):
        if isinstance(x, MetalArray):
            return x if dtype is None or x.dtype == dtype else self.be.adopt(x._a.astype(dtype))
        if emu._is_host_array(x):
            raise ResidencyError("xp.asarray of a host array: use Backend.to_device")
        return self.be.adopt(np.asarray(x, dtype=dtype))

    def ascontiguousarray(self, x, dtype=None):
        if isinstance(x, MetalArray) and x._a.flags.c_contiguous and (dtype is None or x.dtype == dtype):
            return x
        self.be.cpu_touch("xp:ascontiguousarray")
        return self.be.adopt(np.ascontiguousarray(emu._unwrap(x), dtype=dtype))


# ------------------------------------------------------------------ the backend
class _Runtime:
    """The process's one Metal device, arena and queue, shared by every MetalBackend."""

    def __init__(self, arena_bytes):
        if arena_bytes >= (1 << 31) * 4:
            raise ValueError("pydyn.gpu.metal: the kernels index in 32 bits -- an arena under 8 GB")
        self.L = _shim()
        host, gpu, name = ctypes.c_void_p(), ctypes.c_uint64(), ctypes.create_string_buffer(256)
        rc = self.L.pdm_init(arena_bytes, ctypes.byref(host), ctypes.byref(gpu), name, 256)
        if rc:
            raise RuntimeError(f"pydyn.gpu.metal: no Metal device (pdm_init {rc})")
        self.device_name = name.value.decode()
        self.arena = _Arena(host.value, gpu.value, arena_bytes, self.queued)
        self._progress = (ctypes.c_long(), ctypes.c_long())
        self.err = ctypes.create_string_buffer(4096)
        self.deferred = []  # (flag, exception) checked once the queue has run (Backend.check)
        self.pending = 0  # dispatches queued since the last sync
        self.syncs = 0
        self.sync_seconds = 0.0

    def sync(self):
        if self.pending:
            t = time.perf_counter()
            if self.L.pdm_sync(self.err, len(self.err)):
                raise RuntimeError(f"pydyn.gpu.metal: {self.err.value.decode()}")
            self.pending = 0
            self.syncs += 1
            self.sync_seconds += time.perf_counter() - t
        self.arena.collect()
        if self.deferred:
            failed = [exc for flag, exc in self.deferred if int(flag.base_array[0])]
            self.deferred.clear()
            if failed:
                raise failed[0]

    def gpu_seconds(self):
        s, d, b = ctypes.c_double(), ctypes.c_long(), ctypes.c_long()
        self.L.pdm_stats(ctypes.byref(s), ctypes.byref(d), ctypes.byref(b))
        return s.value

    def progress(self):
        """(command buffers committed, retired): what is queued now goes in committed + 1."""
        c, r = self._progress
        self.L.pdm_progress(ctypes.byref(c), ctypes.byref(r))
        return c.value, r.value

    def queued(self):
        return self.progress()[0]

    def mark(self):
        return self.L.pdm_mark()

    def wait(self, mark):
        if self.L.pdm_wait(mark, self.err, len(self.err)):
            raise RuntimeError(f"pydyn.gpu.metal: {self.err.value.decode()}")
        self.arena.collect(self.progress()[1])


_RUNTIME = None


def runtime():
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _Runtime(_ARENA)
    return _RUNTIME


class MetalBackend(Backend):
    name = "metal"
    real = np.float32
    asynchronous = True

    def __init__(self):
        self.rt = runtime()
        self.L, self.arena, self.device_name = self.rt.L, self.rt.arena, self.rt.device_name
        self._pipelines = {}
        self._defines = {}  # library -> the defines it was built with
        self.cpu_ops = defaultdict(int)
        self.profile = None  # {kernel: [count, gpu seconds]} when set: one command buffer a dispatch
        MetalArray.be = self
        self.xp = _Namespace(self)
        super().__init__()
        self._deferred = self.rt.deferred  # shared: one queue

    @property
    def syncs(self):
        return self.rt.syncs

    # -- arrays
    def is_device(self, x):
        return isinstance(x, MetalArray)

    def adopt(self, x):
        """A NumPy result as a device array: in the arena already (a view), else copied in."""
        if isinstance(x, np.ndarray):
            if self.arena.contains(x):
                return MetalArray(x)
            a = self.arena.array(x.shape, x.dtype)
            a[...] = x
            return MetalArray(a)
        if isinstance(x, tuple):
            return tuple(self.adopt(v) for v in x)
        if isinstance(x, list):
            return [self.adopt(v) for v in x]
        return x

    def cpu_touch(self, kind):
        self.cpu_ops[kind] += 1
        self.rt.sync()

    def _upload(self, host):
        a = self.arena.array(host.shape, host.dtype)
        a[...] = host
        return MetalArray(a)

    def _download(self, dev):
        if not self.settled:  # settled: written before a mark waited for
            self.rt.sync()
        return np.array(dev.base_array, copy=True)

    def _write(self, dev, host):  # shared memory: the CPU writes it now (put's inbox is free)
        dev.base_array[...] = host

    def address(self, a):
        try:
            return a._gpu
        except AttributeError:
            pass
        off = a._a.__array_interface__["data"][0] - self.arena.base
        if not 0 <= off < self.arena.size:
            raise ResidencyError("a kernel argument outside the arena")
        a._gpu = self.arena.gpu + off
        return a._gpu

    # -- kernels
    launch_defines: ClassVar = {}  # extra defines for every module (profiling builds)

    def module(self, **defines):
        return super().module(**{**defines, **self.launch_defines})

    def _compile(self, defines):
        path = _metallib(metal_source(self.signatures, defines))
        lib = self.L.pdm_library(str(path).encode(), self.rt.err, len(self.rt.err))
        if not lib:
            raise RuntimeError(f"pydyn.gpu.metal: {self.rt.err.value.decode()}")
        self._defines[lib] = dict(defines)
        return lib

    def _pipeline(self, lib, kernel):
        key = (lib, kernel)
        if key not in self._pipelines:
            w, mt = ctypes.c_int(), ctypes.c_int()
            p = self.L.pdm_pipeline(
                lib, kernel.encode(), ctypes.byref(w), ctypes.byref(mt), self.rt.err, len(self.rt.err)
            )
            if not p:
                raise RuntimeError(f"pydyn.gpu.metal: {kernel}: {self.rt.err.value.decode()}")
            self._pipelines[key] = (p, min(mt.value, int(os.environ.get("PYDYN_METAL_TG", "64"))))
        return self._pipelines[key]

    def _launch(self, lib, kernel, n, sig, args):
        name = kernel
        if kernel in ROUTED:
            if kernel == "pd_convec" and int(args[4]) <= 32:  # convec_gen's passes, either inverse
                return self._convec_passes(self._defines[lib], n, sig, args)
            if kernel == "pd_inverse_rows" and int(args[1]) <= 32:  # the LAPACK kind's pass alone
                return self._inverse_pass(self._defines[lib], sig, args)
            if lines.enabled(kernel, "metal"):  # a SIMD group a line (csrc/advect_coop.cuh)
                n, faces = lines.launch_shape(kernel, sig, args)
                lib, name = self.module(**self._defines[lib], PD_COOP_FACES=faces), f"{kernel}_coop"
            elif kernel == "pd_convec" and args[6] == 0:  # Crout above 32 levels: convec_metal.h's
                name, n = "pd_convec_simd", -(-n // _CONVEC_CPT) * 32 * _CONVEC_CPT
        pso, tg = self._pipelines.get((lib, name)) or self._pipeline(lib, name)
        if name != kernel:
            tg = 32 * (lines.LINES if name.endswith("_coop") else _CONVEC_CPT)
        self._dispatch(pso, _pack(sig, args, self.address), n, tg, kernel)

    def _scratch(self, shape, dtype, fill=None):
        a = self.arena.array(shape, dtype)
        if fill is not None:
            a[...] = fill
        return MetalArray(a)

    def _pass(self, mod, name, n, ncol, spec, args):
        pso, _ = self._pipeline(mod, f"pd_{name}_{n}")
        self._dispatch(pso, _pack(spec, args, self.address), ncol * 32, 128, f"pd_{name}")

    def _dispatch(self, pso, blob, n, tg, label):
        self.L.pdm_dispatch(pso, blob, len(blob), n, tg)
        self.rt.pending += 1
        if self.profile is not None:
            self._labels.append(label)

    def profile_begin(self, cap=2048):
        """Time every dispatch on the GPU (an encoder each, timestamps at its ends) until
        profile_end, which returns {label: [count, seconds]}."""
        self.sync()
        if self.L.pdm_profile_begin(cap):
            raise RuntimeError("pydyn.gpu.metal: no GPU timestamp counters (or cap too large)")
        self.profile, self._labels = {}, []

    def profile_end(self):
        self.sync()
        out = (ctypes.c_double * len(self._labels))()
        m = self.L.pdm_profile_end(out, len(self._labels))
        prof = {}
        for label, t in zip(self._labels[:m], out, strict=False):  # out has room for every label
            rec = prof.setdefault(label, [0, 0.0])
            rec[0] += 1
            rec[1] += t
        self.profile = None
        return prof

    def sync(self):
        self.rt.sync()

    def queued(self):
        return self.rt.queued()

    def retired(self):
        return self.rt.progress()[1]

    def mark(self):
        return self.rt.mark()

    def wait(self, mark):
        self.rt.wait(mark)

    def gpu_seconds(self):
        return self.rt.gpu_seconds()
