"""The line kernels with a SIMD group / warp a line (csrc/advect_coop.cuh): each dialect's kernel
around a body, and what a launcher needs to dispatch it in the serial kernel's place.

`<kernel>_coop` takes the serial kernel's arguments (Metal: its `<kernel>_args` struct; CUDA:
its parameters) and runs `<kernel>_line` on LINES lines of a group; the group's memory holds
one tracer's four face arrays of PD_COOP_FACES each (a define per launch, the longest line's
faces rounded up to the lanes).  The host builds (emu, emu32) keep the serial kernels: the
tests hold the two bitwise.
"""

import os
import re

from .backend import CSRC, ROUTED

__all__ = ["COOP", "DEFAULT", "LINES", "enabled", "launch_shape", "source"]

LINES = 4  # lines a group: 4 x 32 = 128 threads

# serial kernel -> (the parameters whose product is its line count, a line's longest length
# from the arguments: its faces are that + 1)
COOP = {
    "pd_x_rows": (("jm", "lm"), lambda v: v["m"].shape[0]),  # nrow <= m's im
    "pd_y_steps": (("im", "lm"), lambda v: v["jm"]),
    "pd_z_steps": (("im", "jm"), lambda v: v["lm"]),
    "pd_x_child_rows": (("J", "L"), lambda v: v["n"]),
    "pd_y_child_steps": (("I", "L"), lambda v: v["n"]),
    "pd_y_search": (("im", "lm"), lambda v: v["jm"]),
    "pd_z_search": (("im", "jm"), lambda v: v["lm"]),
    "pd_z_scope_search": (("im", "jm"), lambda v: v["lm"]),
    "pd_held_search": (("I", "L"), lambda v: v["n"]),
}
assert set(COOP) <= ROUTED

_PRELUDE = {
    "metal": "#define PD_SH threadgroup\n"
    "#define PD_LINE_SYNC simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)\n"
    "#define PD_LINE_ALL(b) simd_all((b) != 0)\n"
    "#define PD_LINE_MAX(v) simd_max(v)\n",
    "cuda": "#define PD_SH\n#define PD_LINE_SYNC pd_group_sync()\n"
    "#define PD_LINE_ALL(b) pd_all((b) != 0)\n"
    "static __device__ __forceinline__ real pd_line_max(real v)\n"
    "{\n    for (int d = 16; d > 0; d >>= 1) v = fmax(v, pd_shfl_xor(v, d));\n"
    "    return v;\n}\n"
    "#define PD_LINE_MAX(v) pd_line_max(v)\n",
}


# the ones a backend runs cooperatively by default: CUDA every one (an A100 holds ~220 000
# threads, the serial kernels' lines are 3 000 - 11 000); Metal those an M4 runs faster so
# (it holds ~20 000: x and y rows are already enough, and pay the barriers)
DEFAULT = {
    "cuda": set(COOP),
    "metal": {"pd_z_steps", "pd_x_child_rows", "pd_y_search", "pd_held_search"},
}


def enabled(kernel, backend):
    """Whether `kernel` goes cooperative on `backend` (its name): DEFAULT's choice, or every
    one with PYDYN_COOP=all, none with PYDYN_COOP=0 (for timing either way)."""
    mode = os.environ.get("PYDYN_COOP", "1")
    if kernel not in COOP or mode == "0":
        return False
    return mode == "all" or kernel in DEFAULT.get(backend, ())


def launch_shape(kernel, sig, args):
    """(threads, PD_COOP_FACES) of the cooperative launch of `kernel` with these arguments."""
    v = {name: a for (_, name), a in zip(sig, args)}
    lines_of, length = COOP[kernel]
    nlines = 1
    for p in lines_of:
        nlines *= int(v[p])
    faces = int(length(v)) + 1
    return -(-nlines // LINES) * LINES * 32, -(-faces // 32) * 32


def _param(ctype, name):
    return f"{ctype[:-1].strip()} *{name}" if ctype.endswith("*") else f"{ctype} {name}"


def source(signatures, dialect):
    """The bodies and their `<kernel>_coop` kernels in `dialect` ("metal": appended to
    metal_source's translation unit; "cuda": appended to all.cu)."""
    body = (CSRC / "advect_coop.cuh").read_text()
    if dialect == "metal":
        body = re.sub(r'^#include "[^"]+"\n', "", body, flags=re.M)  # advect.cuh is in the unit already
    out = [_PRELUDE[dialect], body]
    for kernel, (lines_of, _) in COOP.items():
        sig = signatures[kernel]
        mem = f"{LINES}][4 * PD_COOP_FACES"
        if dialect == "metal":
            nlines = " * ".join(f"a.{p}" for p in lines_of)
            call = ", ".join(f"a.{p}" for _, p in sig)
            out.append(
                f"kernel void {kernel}_coop(constant {kernel}_args &a [[buffer(0)]],\n"
                "    uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],\n"
                "    uint ln [[thread_index_in_simdgroup]])\n{\n"
                f"    threadgroup real sh[{mem}];\n"
                f"    i64 line = (i64)gid * {LINES} + sg;\n"
                f"    if (line >= {nlines}) return; /* uniform over the SIMD group */\n"
                f"    {kernel}_line(line, ln, sh[sg], {call});\n}}\n"
            )
        else:
            nlines = " * ".join(lines_of)
            params = ", ".join(_param(c, p) for c, p in sig)
            call = ", ".join(p for _, p in sig)
            out.append(
                f'extern "C" __global__ void {kernel}_coop({params})\n{{\n'
                f"    __shared__ real sh[{mem}];\n"
                f"    i64 line = (i64)blockIdx.x * {LINES} + threadIdx.x / 32;\n"
                "    unsigned ln = threadIdx.x % 32;\n"
                f"    if (line >= {nlines}) return; /* uniform over the warp */\n"
                f"    {kernel}_line(line, ln, sh[threadIdx.x / 32], {call});\n}}\n"
            )
    return "\n".join(out)
