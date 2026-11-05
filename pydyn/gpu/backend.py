"""The backends of pydyn.gpu and the ledger of what crossed between host and device.

    cuda   CuPy arrays; csrc/all.cu compiled by NVRTC with --fmad=false (no contraction), the
           default IEEE division and square root.  Needs an NVIDIA GPU with FP64 and CuPy --
           or an AMD one and CuPy built for HIP: hiprtc with -ffp-contract=off (pd.h's PD_HIP).
    emu    EmuArray (emu.py) arrays in host memory; the same csrc/all.cu compiled by the host
           compiler with -ffp-contract=off and its kernels run as plain loops.  What lets the
           device code be checked bitwise against the NumPy reference on a machine with no GPU.
    metal  float32 (Apple GPUs have no double): metal.py, csrc/ compiled offline by xcrun metal
           with -ffp-contract=off, arrays in one shared-storage arena.
    emu32  the float32 build of csrc/ on the host, the CPU's FZ bit set as the GPU flushes
           subnormals (pd.h): what `metal` is checked bitwise against.

Each pair runs the same IEEE operations in the same order -- the kernels have no reductions
whose order depends on scheduling, only order-free max / or -- so emu's bits are cuda's and
emu32's are metal's.  The float32 pair is not the reference's arithmetic: how far it drifts
from the float64 reference is measured, not assumed (docs/gpu_port.md).

Every transfer goes through `to_device` / `to_host` with a tag, and the `Ledger` counts bytes by
direction and tag.  A run's state is uploaded once ("seed"); model inputs read from files
("input") go up when read; nothing of the state comes down except for a comparison
("verify").  `Ledger.state_downloads` is what a test asserts to be zero.
"""

import ctypes
import os
import platform
import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import ClassVar

import numpy as np

from .emu import EmuArray, emu_namespace

__all__ = ["Backend", "Ledger", "Uncapturable", "backend", "set_backend"]

CSRC = Path(__file__).parent / "csrc"
SOURCES = (
    "pd.h",
    "advect.cuh",
    "convec.cuh",
    "mass.cuh",
    "pmath.cuh",
    "chem.cuh",
    "redgrid.cuh",
    "kzz.cuh",
    "child.cuh",
    "boxes.cuh",
    "all.cu",
    "host_twin.inc",
    "convec_metal.h",
)
_THREADS = 128  # threads per block; the kernels are grid-stride loops, any count is correct
# (but convec_gen's pd_cvf divides its shared memory among CVF_WARPS warps a block)
FACE_KERNELS = {"pd_x_rows", "pd_y_steps", "pd_z_steps", "pd_x_child_rows", "pd_y_child_steps"}
# the kernels a device's _launch may run otherwise: convec_gen's passes, coop.COOP's line kernels
ROUTED = {"pd_convec", "pd_inverse_rows", *FACE_KERNELS}
ROUTED |= {"pd_y_search", "pd_z_search", "pd_z_scope_search", "pd_held_search"}


def _source_text():
    return "".join((CSRC / name).read_text() for name in SOURCES)


def kernel_signatures():
    """{kernel: [(ctype, name), ...]} parsed from the PD_GLOBAL declarations of csrc/."""
    text = re.sub(r"/\*.*?\*/", "", _source_text(), flags=re.S)
    sigs = {}
    for name, params in re.findall(r"PD_GLOBAL\s+void\s+(\w+)\s*\(([^)]*)\)", text):
        out = []
        for p in params.split(","):
            p = " ".join(p.replace("PD_G ", "").split())
            ctype, pname = p.rsplit(" ", 1)
            while pname.startswith("*"):
                ctype, pname = ctype + "*", pname[1:]
            out.append((ctype.replace(" *", "*"), pname))
        sigs[name] = out
    return sigs


_POINTEE = {"int": np.int32, "unsigned char": np.uint8, "i64a": np.int64}


def _pointee(ctype, real):
    base = ctype.replace("const ", "").rstrip("*").strip()
    return real if base == "real" else _POINTEE[base]


class Ledger:
    """Bytes and transfers by (direction, tag)."""

    def __init__(self):
        self.bytes = defaultdict(int)
        self.count = defaultdict(int)

    def add(self, direction, tag, nbytes):
        self.bytes[direction, tag] += int(nbytes)
        self.count[direction, tag] += 1

    @property
    def state_downloads(self):
        """Device -> host bytes for anything but a comparison."""
        return sum(b for (d, tag), b in self.bytes.items() if d == "d2h" and tag != "verify")

    def report(self):
        rows = sorted(self.bytes)
        return "\n".join(
            f"  {d} {tag:<8} {self.count[d, tag]:>8} transfers {self.bytes[d, tag] / 1e6:12.3f} MB"
            for d, tag in rows
        )

    def reset(self):
        self.bytes.clear()
        self.count.clear()


class Backend:
    """Common to both: transfers, element strides, the kernel launcher."""

    name = None
    real = np.float64  # what csrc's `real` is in this backend's build

    def __init__(self):
        self.ledger = Ledger()
        self._modules = {}
        self._deferred = []
        self._assumptions = []
        self.signatures = kernel_signatures()
        self._convec_kept = {}  # convec_gen's A, Y, inputs copy and flags, per columns, lmax and kind
        self._plans = {}  # kernel -> launch's checks (_plan)
        self._inboxes = {}  # put's: (view address, shape, strides) -> [[inbox, its COPY, used], ...]
        self._copies = {}  # contiguous's: (view address, shape, strides) -> (scratch, its COPY)

    # -- transfers
    def to_device(self, host, tag="input", dtype=None):
        if dtype is None and np.asarray(host).dtype.kind == "f":
            dtype = self.real
        host = np.ascontiguousarray(host, dtype=dtype)
        self.ledger.add("h2d", tag, host.nbytes)
        return self._upload(host)

    def scratch(self, shape, dtype=None, fill=0):
        """An interval's scratch array, filled: fresh, or inside a speculated call (zoom.Pipeline)
        its slot's arena's, so that the same request of every interval gets the same address
        (what replaying a captured interval needs).  Only for arrays nothing reads once the
        call has been checked."""
        dtype = self.real if dtype is None else dtype
        if self.slot_arena is not None:
            return self.slot_arena.take(shape, dtype, fill)
        if fill:
            return self.to_device(np.full(shape, fill, dtype), "control")
        return self.xp.zeros(shape, dtype=dtype)

    slot_arena = None  # zoom.Pipeline's slot's, while it queues a call (arena_for)

    def arena_for(self, key):
        """A kept arena for zoom.Pipeline's slot `key`, or None: fresh arrays every call."""
        return None

    def to_host(self, dev, tag="verify"):
        host = self._download(dev)
        self.ledger.add("d2h", tag, host.nbytes)
        return host

    # -- error flags and speculation
    asynchronous = False  # whether launches return before the kernel ran
    array_type = None  # the device array class (launch's fast check; is_device decides)
    speculating = False  # inside zoom.speculate: host decisions assumed, checked at its end

    def check(self, flag, exc, deferred=False):
        """Raise exc if the device int flag is set: now, or (deferred, or speculating, on an
        asynchronous backend) at the next sync -- for the advection's fatal errors, so that
        the step need not wait for each kernel; a decision the host takes on a flag is never
        deferred (it is assumed, speculating: assume)."""
        if (deferred or self.speculating) and self.asynchronous:
            self._deferred.append((flag, exc))
        elif int(self.to_host(flag, "control")[0]):
            raise exc

    def assume(self, verdict):
        """Speculating: verdict() -> bool, called after the speculation's sync."""
        self._assumptions.append(verdict)

    def put(self, view, host, tag="input"):
        """view[...] = host, through a counted upload.  On an asynchronous backend the host need
        not wait for the queue: the upload goes to an inbox of the view's (one whose last copy
        has run) and a kept COPY program moves it into the view."""
        host = np.ascontiguousarray(np.broadcast_to(np.asarray(host, dtype=view.dtype), view.shape))
        if not self.asynchronous:
            view[...] = self.to_device(host, tag)
            return
        slots = self._inboxes.setdefault((self.address(view), view.shape, self.strides(view)), [])
        slot = next((s for s in slots if self._inbox_free(s[2])), None)
        if slot is None:
            from .boxes import COPY, Batch, Program

            inbox = self._scratch(view.shape, view.dtype)
            b = Batch()
            b.add(COPY, view, inbox, check=False)  # the inbox is the view's alone
            slot = [inbox, Program([b], keep=(inbox,)), None]
            slots.append(slot)
        self.ledger.add("h2d", tag, host.nbytes)
        self._write(slot[0], host)
        slot[2] = self.queued()
        slot[1].run()

    def _inbox_free(self, used):
        """Whether an inbox whose COPY was queued at queued() == used may be written now."""
        return used is None or self.retired() > used

    # -- the queue's progress (zoom.Pipeline: the device runs a speculation while the host
    # queues the next).  Synchronous backends: everything queued has run.
    settled = False  # zoom.Pipeline's verdicts: a read is of what a waited mark covers

    def queued(self):
        """A tag for what is queued now: it has run once retired() exceeds it."""
        return 0

    def retired(self):
        return 1

    def mark(self):
        """A point in the queue: wait(mark) returns once what was queued before it has run."""
        return 0

    def wait(self, mark):
        pass

    def _write(self, dev, host):
        """dev[...] = host (contiguous, dev's shape and dtype) for put's inbox."""
        raise NotImplementedError

    # -- device arrays
    def is_device(self, x):
        raise NotImplementedError

    def contiguous(self, x):
        """x, or a contiguous copy of it on the device for the next launch: a kept COPY into
        a scratch kept per view, so the host does not wait (the queue runs in order: the next
        call's copy waits for this one's readers)."""
        if x.flags.c_contiguous:
            return x
        key = (self.address(x), x.shape, self.strides(x))
        got = self._copies.get(key)
        if got is None:
            from .boxes import COPY, Batch, Program

            out = self._scratch(x.shape, x.dtype)
            b = Batch()
            b.add(COPY, out, x, check=False)  # out is the view's alone
            got = self._copies[key] = (out, Program([b], keep=(out,)))
        got[1].run()
        return got[0]

    @staticmethod
    def strides(a, nd=None):
        """Element strides of a device view, padded with zeros to nd."""
        s = [st // a.itemsize for st in a.strides]
        return (*s, *([0] * ((nd or len(s)) - len(s))))

    # -- convec_gen's passes (Metal, CUDA): pd_convec for lmax <= 32, a SIMD group / warp a column
    CONVEC_PASSES: ClassVar = {0: ("cvc", "cvb", "cvl", "cvs", "cvu"), 1: ("cvc", "cvb", "cvg", "cvu")}

    def _convec_passes(self, defines, ncol, sig, args):
        """pd_convec as convec_gen's passes for its lmax and inverse kind (0 Crout, 1 LAPACK).
        A, Y and the copy of the inputs are kept per (columns, mass view, lmax, dt, kind): a
        call whose column inputs are as they were reuses the inverse (pd_cvc decides, per
        column, on the device)."""
        from . import convec_gen

        v = {name: a for (_, name), a in zip(sig, args, strict=True)}
        n, kind = int(v["lmax"]), int(v["kind"])
        mod = self.module(**defines, PD_CONVEC_N=n)
        key = (self.address(v["cols"]), self.address(v["m"]), n, float(v["dt"]), ncol, kind)
        kept = self._convec_kept.get(key)
        if kept is None:
            passes = self.CONVEC_PASSES[kind]
            kept = self._convec_kept[key] = {
                "A": self._scratch((ncol, n, n) if passes != ("cvf",) else (1,), self.real),
                "Y": self._scratch((ncol, n, n), self.real),
                "S": self._scratch(
                    (ncol, 6 * n + 2), self.real, np.nan
                ),  # nothing kept: every column differs
                "F": self._scratch((ncol,), np.int32, 0),
            }
        v.update(kept)
        for name in self.CONVEC_PASSES[kind]:
            spec = convec_gen.ARGS[name]
            self._pass(mod, name, n, ncol, spec, [v[f] for _, f in spec])

    def _inverse_pass(self, defines, sig, args):
        """pd_inverse_rows as the LAPACK kind's pass alone (pd_cvg, every column flagged)."""
        from . import convec_gen

        v = {name: a for (_, name), a in zip(sig, args, strict=True)}
        ncol, n = int(v["ncol"]), int(v["n"])
        v["F"] = self._scratch((ncol,), np.int32, 1)
        spec = convec_gen.ARGS["cvg"]
        self._pass(self.module(**defines, PD_CONVEC_N=n), "cvg", n, ncol, spec, [v[f] for _, f in spec])

    def _scratch(self, shape, dtype, fill=None):
        raise NotImplementedError

    def _pass(self, mod, name, n, ncol, spec, args):
        raise NotImplementedError

    # -- kernels
    def module(self, **defines):
        key = tuple(sorted(defines.items())) if defines else ()
        if key not in self._modules:
            self._modules[key] = self._compile(defines)
        return self._modules[key]

    def launch(self, kernel, n, *args, defines=None):
        """Run `kernel` over n work items; args in the order of its C signature (arrays for
        pointers, None for a null pointer, Python numbers for scalars).  The kernels that hold
        a line's face values per tracer are built for the next power of two of their nt
        (PD_NT_MAX: the per-thread arrays' size, not the arithmetic)."""
        plan = self._plans.get(kernel) or self._plan(kernel)
        sig, pointers, nt = plan
        if len(sig) != len(args):
            raise TypeError(f"{kernel}: {len(args)} arguments for {len(sig)} parameters")
        if nt is not None:
            defines = {**(defines or {}), "PD_NT_MAX": 1 << max(0, int(args[nt]) - 1).bit_length()}
        array = self.array_type
        for i, dtype in pointers:
            a = args[i]
            if a is not None and (type(a) is not array or a.dtype != dtype):
                ctype, pname = sig[i]
                if not self.is_device(a):
                    raise TypeError(f"{kernel}({pname}): not a device array")
                if a.dtype != dtype:
                    raise TypeError(f"{kernel}({pname}): dtype {a.dtype}, the kernel reads {ctype}")
        if n > 0:
            self._launch(self.module(**defines) if defines else self.module(), kernel, int(n), sig, args)

    def _plan(self, kernel):
        """(signature, [(pointer position, its dtype)], nt's position or None), once a kernel."""
        sig = self.signatures[kernel]
        pointers = [(i, np.dtype(_pointee(c, self.real))) for i, (c, _) in enumerate(sig) if c.endswith("*")]
        nt = [name for _, name in sig].index("nt") if kernel in FACE_KERNELS else None
        self._plans[kernel] = plan = (sig, pointers, nt)
        return plan


class Uncapturable(RuntimeError):
    """Work a captured graph would not replay faithfully: an upload, a read back, a sync, a
    mark or a fresh allocation while the stream is being captured."""


class _Arena:
    """A speculated call's scratch: its requests, in order, given the arrays the same requests
    of the slot's last call had (a new one where a shape or dtype differs: `moved`), cleared or
    filled on the stream.  Reused only once that call has been checked (zoom.Pipeline's slot)."""

    def __init__(self, be):
        self.be, self.arrays, self.at, self.moved = be, [], 0, False

    def reset(self):
        self.at, self.moved = 0, False

    def take(self, shape, dtype, fill):
        shape, dtype = (tuple(shape) if np.iterable(shape) else (int(shape),)), np.dtype(dtype)
        a = self.arrays[self.at] if self.at < len(self.arrays) else None
        if a is None or a.shape != shape or a.dtype != dtype:
            if self.be.capturing:
                raise Uncapturable("an arena array allocated (or moved) while capturing")
            a = self.be.cupy.empty(shape, dtype)
            self.arrays[self.at : self.at + 1] = [a]
            self.moved = True
        self.at += 1
        if fill:
            a.fill(fill)
        else:
            a.data.memset_async(0, a.nbytes, self.be.stream)
        return a


class _Staging:
    """Pinned host memory that uploads are copied through, so that cudaMemcpyAsync returns at
    once (from pageable memory it may wait for the stream); a chunk is reused once the copies
    from it have run (its last take's queue tag retired)."""

    CHUNK = 64 << 20

    def __init__(self, cupy):
        self.cupy = cupy
        self.chunks = []  # [pinned memory, its uint8 view, bytes used, its last take's tag]

    def take(self, nbytes, tag, retired):
        """A uint8 view of nbytes of pinned memory nothing queued reads."""
        n = -(-nbytes // 256) * 256
        for c in self.chunks:
            if c[3] is not None and c[3] < retired:
                c[2], c[3] = 0, None
            if c[1].size - c[2] >= n:
                c[2], c[3] = c[2] + n, tag
                return c[1][c[2] - n : c[2] - n + nbytes]
        size = max(self.CHUNK, n)
        mem = self.cupy.cuda.alloc_pinned_memory(size)
        self.chunks.append([mem, np.frombuffer(mem, np.uint8, size), n, tag])
        return self.chunks[-1][1][:nbytes]


class _HipGraph:
    """A captured stream's graph, instantiated: HIP's own calls, CuPy having no stream capture
    on HIP. launch(stream) as cupy.cuda.Graph's."""

    _lib = None

    @classmethod
    def lib(cls):
        if cls._lib is None:
            lib = ctypes.CDLL("libamdhip64.so")  # the one CuPy loaded, by LD_LIBRARY_PATH
            vp, pvp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            for name, args in {
                "hipStreamBeginCapture": (vp, ctypes.c_int),
                "hipStreamEndCapture": (vp, pvp),
                "hipGraphInstantiate": (pvp, vp, vp, vp, ctypes.c_size_t),
                "hipGraphLaunch": (vp, vp),
                "hipGraphExecDestroy": (vp,),
                "hipGraphDestroy": (vp,),
            }.items():
                f = getattr(lib, name)
                f.argtypes, f.restype = args, ctypes.c_int
            cls._lib = lib
        return cls._lib

    @staticmethod
    def check(err):
        if err:
            raise RuntimeError(f"pydyn.gpu: HIP error {err}")

    @classmethod
    def end(cls, stream):
        """The stream's capture ended and instantiated, or None if it broke."""
        lib, graph = cls.lib(), ctypes.c_void_p()
        if lib.hipStreamEndCapture(stream.ptr, ctypes.byref(graph)):
            return None
        g = cls()
        g.graph, g.exec = graph, ctypes.c_void_p()
        if lib.hipGraphInstantiate(ctypes.byref(g.exec), graph, None, None, 0):
            g.exec = None
            return None
        return g

    def launch(self, stream):
        self.check(self.lib().hipGraphLaunch(self.exec, stream.ptr))

    def __del__(self):
        if getattr(self, "exec", None):
            self.lib().hipGraphExecDestroy(self.exec)
        if getattr(self, "graph", None):
            self.lib().hipGraphDestroy(self.graph)


class CudaBackend(Backend):
    """cupy on a stream of its own: launches, the uploads (through _Staging) and put's copies
    queue without the host waiting; a download, a host decision on a flag, or sync waits.
    A mark is an event; a settled read (zoom.Pipeline's verdicts) goes on a second stream, so
    that it does not wait for what was queued after the mark."""

    name = "cuda"
    asynchronous = True
    # the Crout kind as one kernel, the matrix in shared memory (convec_gen._fused)
    CONVEC_PASSES: ClassVar = {0: ("cvf",), 1: ("cvc", "cvb", "cvg", "cvu")}

    def __init__(self):
        import cupy

        self.cupy = cupy
        self.xp = cupy
        self.array_type = cupy.ndarray
        self.hip = cupy.cuda.runtime.is_hip
        self._defines = {}  # id(module) -> the defines it was built with
        self._functions = {}  # (id(module), kernel) -> (its function, the scalars' types)
        self._staging = _Staging(cupy)
        self.stream = cupy.cuda.Stream(non_blocking=True)
        self.stream.use()  # cupy's own operations queue on it too
        self.side = cupy.cuda.Stream(non_blocking=True)
        self._marks = deque()  # (mark, its event), oldest first
        self._queued = self._retired = 0  # marks placed; marks waited for (or covered by a sync)
        self.syncs = 0
        self.profile = None  # [(kernel, start, end)] when set: an event pair a launch
        self._keep = []
        self._arenas = {}  # zoom.Pipeline's slots' scratch (arena_for)
        super().__init__()

    def is_device(self, x):
        return isinstance(x, self.cupy.ndarray)

    def _upload(self, host):
        dev = self.cupy.empty(host.shape, host.dtype)
        self._write(dev, host)
        return dev

    def _write(self, dev, host):  # queued after what the stream holds: an inbox is always free
        if self.capturing:
            raise Uncapturable("an upload while capturing (its staging is reused)")
        if host.nbytes:
            buf = self._staging.take(host.nbytes, self._queued, self._retired)
            buf[...] = host.reshape(-1).view(np.uint8)
            dev.data.copy_from_host_async(ctypes.c_void_p(buf.ctypes.data), host.nbytes, self.stream)

    def _inbox_free(self, used):
        return True

    def _download(self, dev):
        if self.capturing:
            raise Uncapturable("a read back while capturing")
        if self.settled:  # written before a mark waited for: not behind what came after it
            if not dev.flags.c_contiguous:
                raise ValueError("pydyn.gpu: a settled read of a strided array")
            return dev.get(stream=self.side)
        self.sync()
        return self.cupy.asnumpy(dev)

    def queued(self):
        return self._queued

    def retired(self):
        return self._retired

    def mark(self):
        if self.capturing:
            raise Uncapturable("a mark while capturing")
        ev = self.cupy.cuda.Event(disable_timing=True)
        ev.record(self.stream)
        self._queued += 1
        self._marks.append((self._queued, ev))
        return self._queued

    def wait(self, mark):
        while self._marks and self._marks[0][0] <= mark:
            m, ev = self._marks.popleft()
            ev.synchronize()
            self._retired = m

    def address(self, a):
        return int(a.data.ptr)

    def _compile(self, defines):
        from . import convec_gen

        nofma = "-ffp-contract=off" if self.hip else "--fmad=false"
        opts = (nofma, "-std=c++14", f"-I{CSRC}", *(f"-D{k}={v}" for k, v in defines.items()))
        code = (CSRC / "all.cu").read_text()
        if "PD_CONVEC_N" in defines:  # convec_gen's passes for that lmax, as CUDA in double
            code += convec_gen.convec_source(int(defines["PD_CONVEC_N"]), "cuda")
        if "PD_COOP_FACES" in defines:  # the line kernels a warp a line
            from . import coop

            code += coop.source(self.signatures, "cuda")
        mod = self.cupy.RawModule(code=code, options=opts)
        self._defines[id(mod)] = dict(defines)
        return mod

    def _scratch(self, shape, dtype, fill=None):
        a = self.cupy.zeros(shape, dtype=dtype)
        if fill is not None:
            a.fill(fill)
        return a

    def arena_for(self, key):
        return self._arenas.setdefault(key, _Arena(self))

    def _pass(self, mod, name, n, ncol, spec, args):
        self._kernel(mod, f"pd_{name}_{n}", ncol * 32, spec, args)  # a warp a column

    def _launch(self, mod, kernel, n, sig, args):
        if kernel in ROUTED:
            if kernel == "pd_convec" and int(args[4]) <= 32:  # convec_gen's passes, either inverse
                return self._convec_passes(self._defines[id(mod)], n, sig, args)
            if kernel == "pd_inverse_rows" and int(args[1]) <= 32:
                return self._inverse_pass(self._defines[id(mod)], sig, args)
            from . import coop

            if coop.enabled(kernel, "cuda"):  # a warp a line (csrc/advect_coop.cuh)
                threads, faces = coop.launch_shape(kernel, sig, args)
                mod = self.module(**self._defines[id(mod)], PD_COOP_FACES=faces)
                return self._kernel(mod, f"{kernel}_coop", threads, sig, args)
        self._kernel(mod, kernel, n, sig, args)

    def _kernel(self, mod, kernel, n, sig, args):
        """cuLaunchKernel with the kernel's parameter buffer, made once and refilled a call:
        CuPy's call and its argument conversion cost ~20 us of host time a launch, this ~3
        (an A100 node's EPYC) -- the host's issue rate is what bounds a speculated interval."""
        key = (id(mod), kernel)
        got = self._functions.get(key)
        if got is None:
            got = self._functions[key] = self._parameters(mod, kernel, sig)
        fptr, slots, params = got
        for (slot, kind), a in zip(slots, args, strict=True):
            if kind == "ptr":
                slot.value = 0 if a is None else a.data.ptr
            elif kind == "real":
                slot.value = float(a)
            else:
                a = int(a)
                if kind == "int" and not -(2**31) <= a < 2**31:
                    raise OverflowError(f"{kernel}: {a} is not an int")
                slot.value = a
        blocks = max(1, min(-(-n // _THREADS), 2**31 - 1))
        launch = self.cupy.cuda.driver.launchKernel
        if self.profile is None:
            return launch(fptr, blocks, 1, 1, _THREADS, 1, 1, 0, self.stream.ptr, params, 0)
        ev = self.cupy.cuda.Event
        a, b = ev(), ev()
        a.record(self.stream)
        launch(fptr, blocks, 1, 1, _THREADS, 1, 1, 0, self.stream.ptr, params, 0)
        b.record(self.stream)
        self.profile.append((kernel, a, b))

    def _parameters(self, mod, kernel, sig):
        """(the function's address, [(a parameter's ctypes slot, its type)], the address of the
        array of the slots' addresses): what cuLaunchKernel reads, copied at each launch."""
        ctype_of = {"i64": ctypes.c_int64, "int": ctypes.c_int32, "real": ctypes.c_double}
        slots = []
        for ctype, _ in sig:
            if ctype.endswith("*"):
                slots.append((ctypes.c_uint64(0), "ptr"))
            elif ctype in ctype_of:
                slots.append((ctype_of[ctype](0), ctype))
            else:
                raise TypeError(f"{kernel}: parameter type {ctype}")
        params = (ctypes.c_void_p * max(1, len(slots)))(*(ctypes.addressof(s) for s, _ in slots))
        f = mod.get_function(kernel)
        self._keep.append((f, slots, params))  # alive while their addresses are launched with
        return f.kernel.ptr, slots, ctypes.addressof(params)

    def profile_begin(self):
        """Time every launch on the GPU (events at its ends) until profile_end, which returns
        {kernel: [count, seconds]}; cupy's own operations are not in it."""
        self.sync()
        self.profile = []

    def profile_end(self):
        self.sync()
        prof = {}
        for kernel, a, b in self.profile:
            rec = prof.setdefault(kernel, [0, 0.0])
            rec[0] += 1
            rec[1] += self.cupy.cuda.get_elapsed_time(a, b) / 1e3
        self.profile = None
        return prof

    # -- CUDA graphs: zoom.Pipeline's repeatable calls captured once, then replayed
    capturing = False

    def capture_begin(self):
        """Queue nothing until capture_end: the stream's work recorded as a graph instead."""
        if self.hip:
            _HipGraph.check(_HipGraph.lib().hipStreamBeginCapture(self.stream.ptr, 0))  # mode global
        else:
            self.stream.begin_capture(self.cupy.cuda.runtime.streamCaptureModeGlobal)
        self.capturing = True

    def capture_end(self):
        """The graph of what was queued since capture_begin (None if the capture broke)."""
        self.capturing = False
        if self.hip:
            return _HipGraph.end(self.stream)
        try:
            return self.stream.end_capture()
        except self.cupy.cuda.runtime.CUDARuntimeError:
            return None

    def replay(self, graph):
        graph.launch(self.stream)

    def sync(self):
        """Wait for the stream; then raise the first deferred error set (Backend.check)."""
        if self.capturing:
            raise Uncapturable("a sync while capturing")
        self.stream.synchronize()
        self.syncs += 1
        self._queued += 1  # as a mark waited for: everything queued has run
        self._retired = self._queued
        self._marks.clear()
        if self._deferred:
            flags = [f.ravel()[:1] for f, _ in self._deferred]
            set_ = self.cupy.asnumpy(self.cupy.concatenate(flags))
            failed = [exc for (_, exc), v in zip(self._deferred, set_, strict=True) if v]
            self._deferred.clear()
            if failed:
                raise failed[0]


class EmuBackend(Backend):
    name = "emu"
    defines: ClassVar = {}

    array_type = EmuArray

    def __init__(self):
        self.xp = emu_namespace(self.real)
        super().__init__()

    def is_device(self, x):
        return isinstance(x, EmuArray)

    def _upload(self, host):
        return EmuArray(host.copy())

    def _scratch(self, shape, dtype, fill=None):
        return EmuArray(np.zeros(shape, dtype) if fill is None else np.full(shape, fill, dtype))

    def _download(self, dev):
        return np.array(dev.base_array, copy=True)

    def address(self, a):
        return int(a.base_array.__array_interface__["data"][0])

    def _compile(self, defines):
        import ctypes

        # PYDYN_EMU_CONTRACT=1 lets the compiler fuse a*b+c into an FMA: the failure --fmad=false
        # exists to prevent, kept so that tests can show the bitwise checks catch it
        contract = (
            ["-ffp-contract=fast", "-march=native"]
            if os.environ.get("PYDYN_EMU_CONTRACT") == "1"
            else ["-ffp-contract=off"]
        )
        flags = [
            "-O2",
            *contract,
            "-fno-fast-math",
            "-std=c++14",
            "-fPIC",
            "-shared",
            *(f"-D{k}={v}" for k, v in sorted({**self.defines, **defines}.items())),
        ]
        so = host_library(flags)  # keyed on the compiler too
        lib = ctypes.CDLL(str(so))
        return lib

    def _launch(self, lib, kernel, n, sig, args):
        import ctypes

        f = getattr(lib, kernel)
        if f.argtypes is None:
            types = {
                "i64": ctypes.c_longlong,
                "int": ctypes.c_int,
                "real": ctypes.c_double,
            }  # i64: long long here
            if self.real == np.float32:
                types["real"] = ctypes.c_float
            f.argtypes = [ctypes.c_void_p if c.endswith("*") else types[c] for c, _ in sig]
            f.restype = None
        conv = []
        for (ctype, _), a in zip(sig, args, strict=True):
            if ctype.endswith("*"):
                conv.append(None if a is None else a.base_array.ctypes.data)
            else:
                conv.append(a)
        f(*conv)

    def sync(self):
        pass


class Emu32Backend(EmuBackend):
    """csrc/ built with PD_F32 on the host, FZ set per kernel: metal's bitwise twin."""

    name = "emu32"
    real = np.float32
    defines: ClassVar = {"PD_F32": 1}


def host_library(flags=None):
    """csrc/ compiled for the host: the emulation kernels plus the twin dgetrf_ / dgetri_ /
    exp / log / pow (csrc/host_twin.inc), cached (pydyn._cache) by a hash of the sources, the
    flags and the compiler ($CXX and its --version: emu's bitwise result is that compiler's).
    Returns the library's path; a cached file that will not load is built again."""
    import ctypes

    from .._cache import atomic, built, command, compiler_id, key

    if flags is None:
        flags = ["-O2", "-ffp-contract=off", "-fno-fast-math", "-std=c++14", "-fPIC", "-shared"]
    cxx = command("CXX", "c++")
    name = f"_pd_host_{key(_source_text(), ' '.join(flags), platform.platform(), compiler_id(tuple(cxx)))}.so"

    def build(path):
        def make(tmp):
            (tmp / "host.cpp").write_text('#include "all.cu"\n#include "host_twin.inc"\n')
            subprocess.run([*cxx, *flags, f"-I{CSRC}", "-o", str(tmp / "host.so"), str(tmp / "host.cpp")],
                           check=True, capture_output=True, text=True)  # fmt: skip
            return tmp / "host.so"

        atomic(path, make)

    path = built(name, build)
    try:
        ctypes.CDLL(str(path))
    except OSError:
        path.unlink(missing_ok=True)
        path = built(name, build)
    return path


_BACKEND = None


def set_backend(name):
    """'cuda' | 'emu' | 'metal' | 'emu32' (or a Backend); returns it."""
    global _BACKEND
    if isinstance(name, Backend):
        _BACKEND = name
    elif name == "cuda":
        _BACKEND = CudaBackend()
    elif name == "emu":
        _BACKEND = EmuBackend()
    elif name == "emu32":
        _BACKEND = Emu32Backend()
    elif name == "metal":
        from .metal import MetalBackend

        _BACKEND = MetalBackend()
    else:
        raise ValueError(f"pydyn.gpu: unknown backend {name!r}")
    return _BACKEND


def backend():
    if _BACKEND is None:
        set_backend(os.environ.get("PYDYN_DEVICE", "emu"))
    return _BACKEND
